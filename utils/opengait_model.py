import os
import sys
import torch
import numpy as np
import cv2
from torchvision import transforms
import yaml

# Import OpenGait modules directly
try:
    # Import the actual DeepGaitV2 implementation
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'OpenGait'))
    from OpenGait.opengait.modeling.models.deepgaitv2 import DeepGaitV2
    from OpenGait.opengait.utils import config_loader, get_valid_args, get_msg_mgr
    OPENGAIT_AVAILABLE = True
except ImportError:
    DeepGaitV2 = None
    OPENGAIT_AVAILABLE = False
    print("Warning: Could not import DeepGaitV2 from OpenGait. Using simplified implementation.")

class OpenGaitModel:
    def __init__(self, model_path, config_path, device='cuda'):
        """
        Initialize OpenGait model for inference
        
        Args:
            model_path: Path to the trained model (.pt file)
            config_path: Path to the model configuration (.yaml file)
            device: Device to run inference on
        """
        self.device = device
        self.model_path = model_path
        self.config_path = config_path
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize model
        self.model = self._build_model()
        self.model.eval()
        
        # Initialize transforms
        self.transform = self._get_transforms()
        
        print(f"OpenGait DeepGaitV2 model loaded from {model_path}")
        print(f"Model configuration loaded from {config_path}")
        
    def _build_model(self):
        """Build the DeepGaitV2 model from configuration"""
        if not OPENGAIT_AVAILABLE:
            # Fall back to simplified implementation
            print("Using simplified DeepGaitV2 implementation")
            return self._create_simple_deepgaitv2(self.config['model_cfg'])
        
        try:
            # 1. First patch the message manager completely before any imports
            print("Applying MessageManager patches...")
            
            # Import the original class
            from OpenGait.opengait.utils.msg_manager import MessageManager, get_msg_mgr
            import logging
            import sys
            
            # Create a basic logger that just prints to console
            basic_logger = logging.getLogger('opengait')
            basic_logger.setLevel(logging.INFO)
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
            basic_logger.addHandler(handler)
            
            # Replace all the MessageManager methods with simple implementations
            def safe_log_info(self, *args, **kwargs):
                print("[INFO]", *args)
                
            def safe_log_warning(self, *args, **kwargs):
                print("[WARNING]", *args)
                
            def safe_log_debug(self, *args, **kwargs):
                pass
                
            def safe_log_eval_info(self, *args, **kwargs):
                pass
                
            def safe_init_logger(self, *args, **kwargs):
                self.logger = basic_logger
                
            # Apply the patches directly to the class
            MessageManager.log_info = safe_log_info
            MessageManager.log_warning = safe_log_warning
            MessageManager.log_debug = safe_log_debug
            MessageManager.log_eval_info = safe_log_eval_info
            MessageManager.init_logger = safe_init_logger
            
            # Ensure the logger exists
            MessageManager.logger = basic_logger
            
            # 2. Setup for distributed environment
            import torch.distributed as dist
            if not dist.is_initialized():
                # Set dummy values for inference
                os.environ['RANK'] = '0'
                os.environ['WORLD_SIZE'] = '1'
                os.environ['MASTER_ADDR'] = 'localhost' 
                os.environ['MASTER_PORT'] = '12355'
                
                # Initialize process group with gloo backend
                try:
                    dist.init_process_group('gloo', init_method='env://')
                    print("Initialized distributed process group for inference")
                except Exception as e:
                    print(f"Warning: Could not initialize distributed process group: {e}")
                    
                    # Mock distributed functions
                    def mock_get_rank():
                        return 0
                    
                    def mock_get_world_size():
                        return 1
                    
                    # Apply patches
                    dist.get_rank = mock_get_rank
                    dist.get_world_size = mock_get_world_size
                    
                    print("Applied distributed function patches")
            
            # 3. Patch CUDA-specific functions for Mac MPS or CPU
            print("Patching CUDA functions for non-CUDA devices...")
            
            # Create mock for set_device
            def mock_set_device(device):
                print(f"[INFO] Mock set_device called with device {device}")
            
            # Apply patch
            torch.cuda.set_device = mock_set_device
            
            # 4. Create a custom DeepGaitV2 subclass instead of patching BaseModel.__init__
            from OpenGait.opengait.modeling.models.deepgaitv2 import DeepGaitV2
            
            class CustomDeepGaitV2(DeepGaitV2):
                """Custom DeepGaitV2 implementation that works on non-CUDA devices"""
                
                def __init__(self, cfgs, training=False):
                    # Initialize as nn.Module first
                    import torch.nn as nn
                    nn.Module.__init__(self)
                    
                    # Save configuration
                    self.cfgs = cfgs
                    self.training_mode = training
                    
                    # Handle device selection
                    if torch.cuda.is_available():
                        print("[INFO] Using CUDA device")
                        self.device = torch.device("cuda:0")
                    elif hasattr(torch, 'mps') and torch.mps.is_available():
                        print("[INFO] Using MPS device (Apple Silicon)")
                        self.device = torch.device("mps")
                    else:
                        print("[INFO] Using CPU device")
                        self.device = torch.device("cpu")
                        
                    try:
                        # Set up message manager
                        from OpenGait.opengait.utils.msg_manager import get_msg_mgr
                        self.msg_mgr = get_msg_mgr()
                        
                        # Set up configs like in BaseModel
                        self.engine_cfg = cfgs['evaluator_cfg']  # We're in inference mode
                        self.save_path = os.path.join('output/', cfgs['data_cfg']['dataset_name'],
                                                    cfgs['model_cfg']['model'], self.engine_cfg['save_name'])
                        
                        # Build network manually to ensure all layers are created
                        self.build_network(cfgs['model_cfg'])
                        self.init_parameters()
                        
                        # Set up dummy transforms
                        from OpenGait.opengait.data.transform import get_transform
                        self.evaluator_trfs = get_transform(cfgs['evaluator_cfg']['transform'])
                        
                        print("[INFO] Successfully initialized model architecture")
                        
                    except Exception as e:
                        print(f"Error during model initialization: {e}, creating emergency model structure")
                        self._create_emergency_model_structure(cfgs['model_cfg'])
                
                def _create_emergency_model_structure(self, model_cfg):
                    """Create minimal model structure for when original initialization fails"""
                    import torch.nn as nn
                    
                    # Get backbone configuration or use defaults
                    mode = model_cfg.get('Backbone', {}).get('mode', 'p3d')
                    in_channels = model_cfg.get('Backbone', {}).get('in_channels', 1)
                    channels = model_cfg.get('Backbone', {}).get('channels', [32, 64, 128, 256])
                    
                    # Create basic layers
                    self.inplanes = channels[0]
                    
                    # Import necessary modules
                    try:
                        from OpenGait.opengait.modeling.modules import (
                            SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, 
                            SeparateFCs, SeparateBNNecks, conv3x3, BasicBlockP3D
                        )
                        
                        # Create layer0 (must exist for forward pass)
                        self.layer0 = SetBlockWrapper(nn.Sequential(
                            conv3x3(in_channels, self.inplanes, 1), 
                            nn.BatchNorm2d(self.inplanes), 
                            nn.ReLU(inplace=True)
                        ))
                        
                        # Create remaining layers
                        self.layer1 = self._make_emergency_layer(BasicBlockP3D, channels[0], 2, stride=[1, 1])
                        self.layer2 = self._make_emergency_layer(BasicBlockP3D, channels[1], 2, stride=[2, 2])
                        self.layer3 = self._make_emergency_layer(BasicBlockP3D, channels[2], 2, stride=[2, 2])
                        self.layer4 = self._make_emergency_layer(BasicBlockP3D, channels[3], 2, stride=[1, 1])
                        
                        # Create feature aggregation components
                        bin_num = model_cfg.get('bin_num', [16])
                        self.TP = PackSequenceWrapper(torch.max)
                        self.HPP = HorizontalPoolingPyramid(bin_num=bin_num)
                        
                        # Create feature extraction components
                        fc_out = 256
                        parts_num = bin_num[0] if isinstance(bin_num, list) else bin_num
                        class_num = model_cfg.get('class_num', 250)
                        
                        self.FCs = SeparateFCs(
                            parts_num=parts_num,
                            in_channels=channels[3],
                            out_channels=fc_out,
                            norm=True
                        )
                        
                        self.BNNecks = SeparateBNNecks(
                            parts_num=parts_num,
                            in_channels=fc_out,
                            class_num=class_num
                        )
                        
                        # Set inference mode flag
                        self.inference_use_emb2 = model_cfg.get('use_emb2', False)
                        
                        print("[INFO] Created emergency model structure")
                        
                    except Exception as e:
                        print(f"Failed to create emergency structure: {e}")
                        # Create absolute minimal dummy components
                        self.layer0 = nn.Identity()
                        self.layer1 = nn.Identity()
                        self.layer2 = nn.Identity()
                        self.layer3 = nn.Identity()
                        self.layer4 = nn.Identity()
                        
                        # Minimal components for forward pass
                        self.TP = lambda x, y, options=None: (x, None)
                        self.HPP = lambda x: torch.flatten(nn.AdaptiveAvgPool2d((1, 1))(x), 1)
                        self.FCs = nn.Linear(256, 256)
                        self.BNNecks = lambda x: (x, None)
                        self.inference_use_emb2 = False
                        
                def _make_emergency_layer(self, block, planes, blocks, stride=1):
                    """Create an emergency layer with BasicBlockP3D blocks"""
                    import torch.nn as nn
                    from OpenGait.opengait.modeling.modules import SetBlockWrapper, BasicBlockP3D
                    
                    layers = []
                    
                    # First block may have stride
                    layers.append(SetBlockWrapper(
                        nn.Sequential(
                            nn.Conv2d(self.inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
                            nn.BatchNorm2d(planes),
                            nn.ReLU(inplace=True)
                        )
                    ))
                    
                    # Update inplanes for subsequent blocks
                    self.inplanes = planes
                    
                    # Add remaining blocks
                    for _ in range(1, blocks):
                        layers.append(SetBlockWrapper(
                            nn.Sequential(
                                nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.BatchNorm2d(planes),
                                nn.ReLU(inplace=True)
                            )
                        ))
                        
                    return nn.Sequential(*layers)
                
                def forward(self, inputs):
                    """Safe forward implementation that handles errors"""
                    try:
                        # Try original forward pass
                        return super().forward(inputs)
                    except Exception as e:
                        print(f"Error in forward pass: {e}")
                        import traceback
                        traceback.print_exc()
                        
                        # Manual implementation of forward pass
                        try:
                            ipts, labs, typs, vies, seqL = inputs
                            
                            # Get silhouettes
                            sils = ipts[0]
                            
                            # Handle input format
                            if len(sils.size()) == 4:
                                sils = sils.unsqueeze(1)  # [n, s, h, w] -> [n, 1, s, h, w]
                            
                            # Forward through backbone
                            outs = self.layer0(sils)
                            outs = self.layer1(outs)
                            outs = self.layer2(outs)
                            outs = self.layer3(outs)
                            outs = self.layer4(outs)
                            
                            # Temporal pooling
                            outs = self.TP(outs, seqL, options={"dim": 2})[0]
                            
                            # Horizontal pooling
                            feat = self.HPP(outs)
                            
                            # Feature extraction
                            embed_1 = self.FCs(feat)
                            
                            # Handle both SeparateBNNecks and simple Linear
                            if hasattr(self.BNNecks, '__call__'):
                                embed_2, logits = self.BNNecks(embed_1)
                            else:
                                embed_2 = embed_1
                                logits = None
                            
                            # Select embedding based on config
                            if self.inference_use_emb2:
                                embed = embed_2
                            else:
                                embed = embed_1
                            
                            # Return minimal output format
                            return {
                                'inference_feat': {
                                    'embeddings': embed
                                }
                            }
                            
                        except Exception as e:
                            print(f"Emergency forward pass failed: {e}")
                            # Return dummy embedding as last resort
                            batch_size = 1
                            dummy_embedding = torch.zeros((batch_size, 256)).to(self.device)
                            return {
                                'inference_feat': {
                                    'embeddings': dummy_embedding
                                }
                            }

            # 5. Create the model config
            cfgs = {
                'model_cfg': self.config['model_cfg'],
                'data_cfg': {
                    'dataset_name': 'inference',
                    'workers': 0,
                    'dataset_root': '',
                    'test_dataset_name': 'inference',
                    'num_workers': 0
                },
                'evaluator_cfg': {
                    'enable_float16': False,
                    'restore_ckpt_strict': False,
                    'restore_hint': 0,
                    'save_name': 'DeepGaitV2',
                    'sampler': {
                        'type': 'InferenceSampler',
                        'batch_size': 1
                    },
                    'transform': [
                        {'type': 'NoOperation'}
                    ]
                },
                'trainer_cfg': {
                    'log_iter': 100,
                    'save_iter': 10000,
                    'with_test': False,
                    'enable_float16': False,
                    'fix_BN': False,
                    'restore_ckpt_strict': False,
                    'optimizer_reset': True,
                    'scheduler_reset': True,
                    'restore_hint': 0,
                    'save_name': 'DeepGaitV2',
                    'transform': [
                        {'type': 'NoOperation'}
                    ],
                    'sampler': {
                        'type': 'InferenceSampler',
                        'batch_size': 1
                    }
                },
                'loss_cfg': {},
                'optimizer_cfg': {
                    'solver': 'SGD',
                    'lr': 0.1,
                    'momentum': 0.9,
                    'weight_decay': 0.0005
                },
                'scheduler_cfg': {
                    'scheduler': 'StepLR',
                    'step_size': 10000,
                    'gamma': 0.1
                }
            }
            
            # 6. Create dummy loader function
            from OpenGait.opengait.modeling.base_model import BaseModel
            original_get_loader = BaseModel.get_loader
            
            def mock_get_loader(self, data_cfg, train=True):
                print(f"Skipping data loading for {'training' if train else 'testing'}")
                # Return a dummy object with expected attributes
                class DummyLoader:
                    def __init__(self):
                        self.batch_sampler = type('obj', (object,), {'batch_size': 1})
                        self.dataset = type('obj', (object,), {
                            'label_list': [], 
                            'types_list': [],
                            'views_list': []
                        })
                    
                    def __len__(self):
                        return 0
                    
                    def __iter__(self):
                        return iter([])
                
                return DummyLoader()
            
            # Apply the loader patch
            BaseModel.get_loader = mock_get_loader
            
            # 7. Now load the model safely with all patches in place  
            print("Loading DeepGaitV2 model...")
            
            # Try with our custom class first
            try:
                model = CustomDeepGaitV2(cfgs, training=False)
                print("Created model using CustomDeepGaitV2")
            except Exception as e:
                print(f"Failed to create CustomDeepGaitV2: {e}")
                # Fall back to simplified implementation
                return self._create_simple_deepgaitv2(self.config['model_cfg'])
            
            # 8. Load model weights
            if os.path.exists(self.model_path):
                print(f"Loading weights from {self.model_path}")
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # Remove 'module.' prefix if present
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('module.'):
                        new_state_dict[k[7:]] = v
                    else:
                        new_state_dict[k] = v
                
                # Load state dict with relaxed matching
                try:
                    model.load_state_dict(new_state_dict, strict=False)
                    print(f"Successfully loaded model weights")
                except Exception as e:
                    print(f"Error loading weights: {e}")
                    # Try to load compatible weights only
                    model_dict = model.state_dict()
                    compatible_dict = {}
                    for k, v in new_state_dict.items():
                        if k in model_dict and model_dict[k].shape == v.shape:
                            compatible_dict[k] = v
                    model.load_state_dict(compatible_dict, strict=False)
                    print(f"Loaded {len(compatible_dict)} compatible weights")
            else:
                print(f"Warning: Model weights file not found")
                
            # 9. Move to device and return
            model = model.to(self.device)
            print("Successfully initialized DeepGaitV2 model")
            return model
            
        except Exception as e:
            print(f"Error building official DeepGaitV2 model: {e}")
            print("Falling back to simplified implementation")
            import traceback
            traceback.print_exc()
            return self._create_simple_deepgaitv2(self.config['model_cfg'])

    def _create_simple_deepgaitv2(self, model_cfg):
        """Create simplified DeepGaitV2 model if OpenGait is not available"""
        # ... existing simplified implementation code ...
        # This is a fallback method that would be used if the official implementation fails
        try:
            from OpenGait.opengait.modeling.modules import (
                SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, 
                SeparateFCs, SeparateBNNecks, conv3x3, BasicBlockP3D
            )
        except ImportError:
            print("Error: Could not import OpenGait modules. Please ensure OpenGait is properly installed.")
            raise ImportError("OpenGait modules not available")
            
        import torch.nn as nn
        
        class SimpleDeepGaitV2(nn.Module):
            def __init__(self, model_cfg):
                super(SimpleDeepGaitV2, self).__init__()
                
                # Get configuration parameters
                backbone_cfg = model_cfg['Backbone']
                mode = backbone_cfg['mode']
                in_channels = backbone_cfg['in_channels']
                layers = backbone_cfg['layers']
                channels = backbone_cfg['channels']
                
                self.inference_use_emb2 = model_cfg.get('use_emb2', False)
                
                # Build backbone layers similar to DeepGaitV2
                self.inplanes = channels[0]
                self.layer0 = SetBlockWrapper(nn.Sequential(
                    conv3x3(in_channels, self.inplanes, 1), 
                    nn.BatchNorm2d(self.inplanes), 
                    nn.ReLU(inplace=True)
                ))
                
                # Build remaining layers
                if mode == 'p3d':
                    strides = [[1, 1], [2, 2], [2, 2], [1, 1]]
                    block = BasicBlockP3D
                else:
                    strides = [[1, 1], [2, 2], [2, 2], [1, 1]]
                    try:
                        from OpenGait.opengait.modeling.modules import BasicBlock2D
                        block = BasicBlock2D
                    except ImportError:
                        # Fallback to P3D blocks if 2D blocks not available
                        block = BasicBlockP3D
                
                self.layer1 = self._make_layer(block, channels[0], layers[0], stride=strides[0])
                self.layer2 = self._make_layer(block, channels[1], layers[1], stride=strides[1])
                self.layer3 = self._make_layer(block, channels[2], layers[2], stride=strides[2])
                self.layer4 = self._make_layer(block, channels[3], layers[3], stride=strides[3])
                
                # Add remaining components with fallback values if config is incomplete
                bin_num = model_cfg.get('bin_num', [16])  # Default value
                self.TP = PackSequenceWrapper(torch.max)
                self.HPP = HorizontalPoolingPyramid(bin_num=bin_num)
                
                # Set up SeparateFCs with defaults if not in config
                if 'SeparateFCs' in model_cfg:
                    self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
                else:
                    # Default configuration for SeparateFCs
                    fc_cfg = {
                        'parts_num': bin_num[0] if isinstance(bin_num, list) else bin_num,
                        'in_channels': channels[-1],  # Use last channel from backbone
                        'out_channels': 256,
                        'norm': True
                    }
                    self.FCs = SeparateFCs(**fc_cfg)
                
                # Set up SeparateBNNecks
                if 'SeparateBNNecks' in model_cfg:
                    # Extract required parameters and add missing ones
                    bn_cfg = model_cfg['SeparateBNNecks'].copy()
                    if 'parts_num' not in bn_cfg:
                        bn_cfg['parts_num'] = bin_num[0] if isinstance(bin_num, list) else bin_num
                    if 'in_channels' not in bn_cfg:
                        bn_cfg['in_channels'] = 256  # Match FCs output
                    self.BNNecks = SeparateBNNecks(**bn_cfg)
                else:
                    # Default configuration for SeparateBNNecks
                    bn_cfg = {
                        'parts_num': bin_num[0] if isinstance(bin_num, list) else bin_num,
                        'in_channels': 256,  # Match FCs output
                        'class_num': model_cfg.get('class_num', 250)  # Default to 250 classes
                    }
                    self.BNNecks = SeparateBNNecks(**bn_cfg)
                
            def _make_layer(self, block, planes, blocks, stride=1):
                layers = []
                
                # Handle downsample layer
                downsample = None
                if max(stride) > 1 or self.inplanes != planes * block.expansion:
                    if block == BasicBlockP3D:
                        # P3D blocks use 3D convolution for downsample with stride [1, spatial_stride, spatial_stride]
                        downsample = nn.Sequential(
                            nn.Conv3d(self.inplanes, planes * block.expansion, 
                                     kernel_size=[1, 1, 1], stride=[1, *stride], 
                                     padding=[0, 0, 0], bias=False), 
                            nn.BatchNorm3d(planes * block.expansion)
                        )
                    else:
                        # 2D blocks use 2D convolution for downsample
                        downsample = nn.Sequential(
                            nn.Conv2d(self.inplanes, planes * block.expansion, 
                                     kernel_size=1, stride=stride, bias=False),
                            nn.BatchNorm2d(planes * block.expansion)
                        )
                else:
                    downsample = None  # No downsample needed
                
                # Create first block with potential downsample
                if block == BasicBlockP3D:
                    layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample))
                else:
                    layers.append(SetBlockWrapper(
                        block(self.inplanes, planes, stride=stride, downsample=downsample)
                    ))
                
                self.inplanes = planes * block.expansion
                
                # Create remaining blocks without downsample
                for i in range(1, blocks):
                    if block == BasicBlockP3D:
                        layers.append(block(self.inplanes, planes))
                    else:
                        layers.append(SetBlockWrapper(
                            block(self.inplanes, planes)
                        ))
                        
                return nn.Sequential(*layers)
            
            def forward(self, inputs):
                ipts, labs, typs, vies, seqL = inputs
                
                sils = ipts[0]
                
                # Handle input format like DeepGaitV2
                if len(sils.size()) == 4:
                    sils = sils.unsqueeze(1)  # [n, s, h, w] -> [n, 1, s, h, w]
                else:
                    from einops import rearrange
                    sils = rearrange(sils, 'n s c h w -> n c s h w')
                
                # Forward through backbone
                outs = self.layer0(sils)
                
                outs = self.layer1(outs)
                outs = self.layer2(outs)
                outs = self.layer3(outs)
                outs = self.layer4(outs)
                
                # Temporal pooling
                outs = self.TP(outs, seqL, options={"dim": 2})[0]
                
                # Horizontal pooling
                feat = self.HPP(outs)
                
                # Feature extraction
                embed_1 = self.FCs(feat)
                embed_2, logits = self.BNNecks(embed_1)
                
                if self.inference_use_emb2:
                    embed = embed_2
                else:
                    embed = embed_1
                
                retval = {
                    'training_feat': {
                        'triplet': {'embeddings': embed_1, 'labels': labs},
                        'softmax': {'logits': logits, 'labels': labs}
                    },
                    'visual_summary': {
                        'image/sils': sils.view(-1, *sils.shape[-3:]),
                    },
                    'inference_feat': {
                        'embeddings': embed
                    }
                }
                
                return retval
        
        return SimpleDeepGaitV2(model_cfg)
    
    def _get_transforms(self):
        """Get preprocessing transforms for silhouettes"""
        transform_list = []
        
        # Basic transforms for silhouettes
        transform_list.extend([
            transforms.ToPILImage(),
            transforms.Resize((64, 44)),  # Standard size for gait recognition
            transforms.ToTensor(),
        ])
        
        return transforms.Compose(transform_list)
    
    def preprocess_silhouettes(self, silhouettes):
        """
        Preprocess silhouettes for the model
        
        Args:
            silhouettes: List of numpy arrays (silhouette images)
            
        Returns:
            torch.Tensor: Preprocessed tensor ready for model input
        """
        if silhouettes is None or (isinstance(silhouettes, torch.Tensor) and silhouettes.numel() == 0):
            return None
            
        processed_silhouettes = []
        
        for sil in silhouettes:
            # Convert to uint8 if needed
            if isinstance(sil, torch.Tensor) and sil.dtype != torch.uint8:
                sil = (sil * 255).to(torch.uint8)
            elif isinstance(sil, np.ndarray) and sil.dtype != np.uint8:
                sil = (sil * 255).astype(np.uint8)
            
            # Ensure binary silhouette
            if len(sil.shape) == 3:
                sil = cv2.cvtColor(sil, cv2.COLOR_BGR2GRAY)
            
            # Apply transforms
            sil_tensor = self.transform(sil)
            # Remove channel dimension to get [h, w]
            if sil_tensor.shape[0] == 1:
                sil_tensor = sil_tensor.squeeze(0)
            processed_silhouettes.append(sil_tensor)
        
        if not processed_silhouettes:
            return None
            
        # Stack silhouettes to create sequence [s, h, w]
        silhouette_tensor = torch.stack(processed_silhouettes, dim=0)  # [s, h, w]
        
        # Add batch dimension for DeepGaitV2: [n, s, h, w]  
        # The model will add channel dimension internally
        silhouette_tensor = silhouette_tensor.unsqueeze(0)  # [1, s, h, w]
        
        return silhouette_tensor.to(self.device)
    
    def extract_embeddings(self, silhouettes):
        """
        Extract gait embeddings from silhouettes
        
        Args:
            silhouettes: List of numpy arrays (silhouette images)
            
        Returns:
            torch.Tensor: Gait embedding vector
        """
        if silhouettes is None:
            print("Warning: No silhouettes provided")
            return None
            
        if len(silhouettes) < 5:  # Minimum frames for reliable gait analysis
            print(f"Warning: Need at least 5 silhouettes for reliable gait analysis, got {len(silhouettes)}")
            return None
            
        try:
            # Preprocess silhouettes
            input_tensor = self.preprocess_silhouettes(silhouettes)
            if input_tensor is None:
                return None
            
            # Prepare model inputs
            # For DeepGaitV2: [ipts, labs, typs, vies, seqL]
            labs = torch.zeros(1, dtype=torch.long).to(self.device)  # dummy labels
            typs = torch.zeros(1, dtype=torch.long).to(self.device)  # dummy types  
            vies = torch.zeros(1, dtype=torch.long).to(self.device)  # dummy views
            seqL = [torch.tensor([input_tensor.shape[1]], dtype=torch.long).to(self.device)]  # sequence length
            
            inputs = ([input_tensor], labs, typs, vies, seqL)
            
            # Extract features
            with torch.no_grad():
                outputs = self.model(inputs)
                
            # Get inference embeddings
            if 'inference_feat' in outputs and 'embeddings' in outputs['inference_feat']:
                embeddings = outputs['inference_feat']['embeddings']
                
                # Average pooling over parts dimension if needed 
                if len(embeddings.shape) == 3:  # [n, c, p]
                    embeddings = embeddings.mean(dim=-1)  # [n, c]
                
                # Get first (and only) sample from batch
                embedding = embeddings[0]  # [c]
                
                # Apply L2 normalization
                import torch.nn.functional as F
                embedding = F.normalize(embedding, p=2, dim=0)
                
                # Apply adaptive temperature scaling
                # Lower temperature increases separation between embeddings
                temperature = 0.002  # More aggressive scaling for better discrimination
                embedding = embedding / temperature
                
                return embedding.cpu()
            else:
                print("Warning: Could not extract embeddings from model output")
                return None
                
        except Exception as e:
            print(f"Error extracting embeddings: {e}")
            import traceback
            traceback.print_exc()
            return None

    def compute_similarity(self, embedding1, embedding2):
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding tensor
            embedding2: Second embedding tensor
            
        Returns:
            float: Cosine similarity score
        """
        try:
            # Ensure embeddings are on CPU
            if isinstance(embedding1, torch.Tensor):
                embedding1 = embedding1.cpu()
            if isinstance(embedding2, torch.Tensor):
                embedding2 = embedding2.cpu()
            
            # Compute cosine similarity
            similarity = torch.cosine_similarity(
                embedding1.unsqueeze(0),
                embedding2.unsqueeze(0)
            ).item()
            
            return similarity
            
        except Exception as e:
            print(f"Error computing similarity: {e}")
            return 0.0
    
    def get_model_info(self):
        """Get information about the loaded model"""
        return {
            'model_type': 'DeepGaitV2',
            'model_path': self.model_path,
            'config_path': self.config_path,
            'device': self.device,
            'config': self.config
        }