import os
import sys
import torch
import numpy as np
import cv2
from torchvision import transforms
import yaml

# Add OpenGait to path
opengait_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'OpenGait')
if opengait_path not in sys.path:
    sys.path.insert(0, opengait_path)

# Import OpenGait modules directly
try:
    from OpenGait.opengait.modeling.models.deepgaitv2 import DeepGaitV2
except ImportError:
    DeepGaitV2 = None
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
        """Build the DeepGaitV2 model from configuration - simplified approach"""
        model_cfg = self.config['model_cfg']
        
        # Debugging: Print model_cfg to verify its structure
        print("Loaded model_cfg:", model_cfg)
        
        # Create a minimal mock cfgs for the model
        mock_cfgs = {
            'model_cfg': model_cfg,
            'trainer_cfg': None,
            'evaluator_cfg': {
                'enable_float16': False,
                'restore_ckpt_strict': True,
                'restore_hint': 0,
                'save_name': 'DeepGaitV2'
            }
        }
        
        # Patch the problematic parts by setting dummy distributed values
        import torch.distributed as dist
        if not dist.is_initialized():
            # Mock distributed setup for inference
            os.environ['RANK'] = '0'
            os.environ['WORLD_SIZE'] = '1'
            os.environ['MASTER_ADDR'] = 'localhost' 
            os.environ['MASTER_PORT'] = '12355'
            
        # Create the DeepGaitV2 model instance
        try:
            # Create model with simplified approach
            model = self._create_simple_deepgaitv2(model_cfg)
        except Exception as e:
            print(f"Error creating DeepGaitV2: {e}")
            raise e
        
        # Load trained weights
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Remove 'module.' prefix if present (from DataParallel training)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            
            # Load state dict with relaxed matching
            try:
                model.load_state_dict(new_state_dict, strict=False)
                print(f"Loaded model weights from {self.model_path}")
            except Exception as e:
                print(f"Error loading weights: {e}")
                # Try to load compatible weights only
                model_dict = model.state_dict()
                compatible_dict = {}
                for k, v in new_state_dict.items():
                    if k in model_dict and model_dict[k].shape == v.shape:
                        compatible_dict[k] = v
                model.load_state_dict(compatible_dict, strict=False)
                print(f"Loaded {len(compatible_dict)} compatible weights from {self.model_path}")
        else:
            print(f"Warning: Model weights file {self.model_path} not found")
            
        # Move model to device
        model = model.to(self.device)
        return model
    
    def _create_simple_deepgaitv2(self, model_cfg):
        """Create DeepGaitV2 model directly without framework overhead"""
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
                        from opengait.modeling.modules import BasicBlock2D
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
                print(f"Input sils shape: {sils.shape}")
                
                # Handle input format like DeepGaitV2
                if len(sils.size()) == 4:
                    sils = sils.unsqueeze(1)  # [n, s, h, w] -> [n, 1, s, h, w]
                    print(f"After unsqueeze: {sils.shape}")
                else:
                    from einops import rearrange
                    sils = rearrange(sils, 'n s c h w -> n c s h w')
                    print(f"After rearrange: {sils.shape}")
                
                # Forward through backbone
                print(f"Before layer0: {sils.shape}")
                outs = self.layer0(sils)
                print(f"After layer0: {outs.shape}")
                
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
            if sil.dtype != torch.uint8:
                sil = (sil * 255).to(torch.uint8)
            
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
        if silhouettes is None or (isinstance(silhouettes, torch.Tensor) and silhouettes.numel() == 0):
            print(f"Warning: Need at least 5 silhouettes for reliable gait analysis, got {len(silhouettes) if silhouettes else 0}")
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
            seqL = [torch.tensor([input_tensor.shape[1]], dtype=torch.long).to(self.device)]  # sequence length from shape[1] (frames dimension)
            
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
