from typing import Optional
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torch import optim
from utils import view_images, aggregate_attention
from distances import LpDistance
import other_attacks


def preprocess(image, res=512):
    image = image.resize((res, res), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)[:, :3, :, :].cuda()
    return 2.0 * image - 1.0


def encoder(image, model, res=512):
    generator = torch.Generator().manual_seed(8888)
    image = preprocess(image, res)
    # Convert to the same dtype as model
    image = image.to(dtype=model.vae.dtype, device=model.device)
    gpu_generator = torch.Generator(device=image.device)
    gpu_generator.manual_seed(generator.initial_seed())
    return 0.13025 * model.vae.encode(image).latent_dist.sample(generator=gpu_generator)


# SDXL的编码方法
def encode_prompt(model, prompt_text: str, device=None):
    """
    为单个文本提示词编码，生成SDXL所需的conditional和unconditional embeddings。
    Args:
        model: SDXL模型
        prompt_text: 单个条件文本提示词 (str)
        device: 设备
    Returns:
        dict: 包含 "prompt_embeds" (cond), "pooled_prompt_embeds" (cond_pooled),
              "uncond_prompt_embeds", "uncond_pooled_prompt_embeds"
    """
    if device is None:
        device = model.device

    # Conditional embeddings
    text_inputs_cond = model.tokenizer(
        [prompt_text],
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    cond_prompt_embeds = model.text_encoder(text_inputs_cond.input_ids.to(device))[0]

    text_inputs_2_cond = model.tokenizer_2(
        [prompt_text],
        padding="max_length",
        max_length=model.tokenizer_2.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    cond_pooled_prompt_embeds = model.text_encoder_2(text_inputs_2_cond.input_ids.to(device))[0]

    # Unconditional embeddings
    uncond_tokens = [""]
    text_inputs_uncond = model.tokenizer(
        uncond_tokens,
        padding="max_length",
        max_length=model.tokenizer.model_max_length, # Use same max_length as conditional
        truncation=True,
        return_tensors="pt",
    )
    uncond_prompt_embeds = model.text_encoder(text_inputs_uncond.input_ids.to(device))[0]

    text_inputs_2_uncond = model.tokenizer_2(
        uncond_tokens,
        padding="max_length",
        max_length=model.tokenizer_2.model_max_length, # Use same max_length as conditional
        truncation=True,
        return_tensors="pt",
    )
    uncond_pooled_prompt_embeds = model.text_encoder_2(text_inputs_2_uncond.input_ids.to(device))[0]
    
    # 确保所有张量使用相同的dtype
    dtype = model.text_encoder.dtype # Assuming text_encoder and text_encoder_2 have same dtype as unet
    cond_prompt_embeds = cond_prompt_embeds.to(dtype=dtype)
    cond_pooled_prompt_embeds = cond_pooled_prompt_embeds.to(dtype=dtype)
    uncond_prompt_embeds = uncond_prompt_embeds.to(dtype=dtype)
    uncond_pooled_prompt_embeds = uncond_pooled_prompt_embeds.to(dtype=dtype)

    print("[ENCODE] cond_prompt_embeds shape:", cond_prompt_embeds.shape)
    print("[ENCODE] cond_pooled_prompt_embeds shape:", cond_pooled_prompt_embeds.shape)
    print("[ENCODE] uncond_prompt_embeds shape:", uncond_prompt_embeds.shape)
    print("[ENCODE] uncond_pooled_prompt_embeds shape:", uncond_pooled_prompt_embeds.shape)
    return {
        "prompt_embeds": cond_prompt_embeds, # Shape: [1, 77, 768]
        "pooled_prompt_embeds": cond_pooled_prompt_embeds, # Shape: [1, 1280]
        "uncond_prompt_embeds": uncond_prompt_embeds, # Shape: [1, 77, 768]
        "uncond_pooled_prompt_embeds": uncond_pooled_prompt_embeds, # Shape: [1, 1280]
    }


@torch.no_grad()
def ddim_reverse_sample(image, cond_prompt_text: str, model, num_inference_steps: int = 20, guidance_scale: float = 2.5, res=512):
    """
    SDXL的DDIM逆向采样过程, 针对单个图像和单个条件提示.
    """
    device = model.device
    dtype = model.unet.dtype # Use unet's dtype as reference for all inputs to unet

    # 1. 文本编码 (conditional and unconditional)
    prompt_data = encode_prompt(model, cond_prompt_text, device=device)
    cond_prompt_embeds = prompt_data["prompt_embeds"]
    cond_pooled_prompt_embeds = prompt_data["pooled_prompt_embeds"]
    uncond_prompt_embeds = prompt_data["uncond_prompt_embeds"]
    uncond_pooled_prompt_embeds = prompt_data["uncond_pooled_prompt_embeds"]

    # 2. 准备time_ids (for a single image concept)
    original_size = (res, res)
    crops_coords_top_left = (0, 0) # Corrected variable name
    target_size = (res, res)
    
    # time_ids for a single instance, will be duplicated for CFG
    single_time_ids = torch.tensor(
        [
            original_size[0], original_size[1],
            crops_coords_top_left[0], crops_coords_top_left[1],
            target_size[0], target_size[1],
        ],
        device=device, dtype=dtype # ensure dtype
    ).unsqueeze(0)  # Shape: [1, 6]

    # 3. 初始化采样过程
    model.scheduler.set_timesteps(num_inference_steps)
    timesteps = model.scheduler.timesteps.flip(0) # DDIM inversion goes from t_0 to t_T-1
    
    latents = encoder(image, model, res=res).to(dtype=dtype) # Shape: [1, 4, H, W]

    all_latents = [latents.clone()] # Store initial latents

    # Debug initial shapes before loop, relevant for CFG setup
    print("[DDIM Pre-Loop Debug] Initial shapes for CFG components:")
    print(f"  latents (single): {latents.shape}")
    print(f"  cond_prompt_embeds (single): {cond_prompt_embeds.shape}")
    print(f"  uncond_prompt_embeds (single): {uncond_prompt_embeds.shape}")
    print(f"  cond_pooled_prompt_embeds (single): {cond_pooled_prompt_embeds.shape}")
    print(f"  uncond_pooled_prompt_embeds (single): {uncond_pooled_prompt_embeds.shape}")
    print(f"  single_time_ids: {single_time_ids.shape}")

    for t in tqdm(timesteps[:-1], desc="DDIM_inverse"):
        # Prepare inputs for CFG (batch size becomes 2)
        latents_input = torch.cat([latents, latents], dim=0).to(dtype=dtype)
        
        # Concatenate unconditional and conditional prompt embeddings
        prompt_embeds_cat = torch.cat([uncond_prompt_embeds, cond_prompt_embeds], dim=0).to(dtype=dtype)
        pooled_prompt_embeds_cat = torch.cat([uncond_pooled_prompt_embeds, cond_pooled_prompt_embeds], dim=0).to(dtype=dtype)
        
        # Duplicate time_ids for CFG
        time_ids_cat = torch.cat([single_time_ids, single_time_ids], dim=0).to(dtype=dtype)

        added_cond_kwargs = {
            "text_embeds": pooled_prompt_embeds_cat,
            "time_ids": time_ids_cat
        }

        # Debug shapes inside loop, right before UNet call
        print("\n[DDIM Loop Debug] Shapes for UNet input:")
        print(f"  latents_input: {latents_input.shape}, dtype: {latents_input.dtype}")
        print(f"  prompt_embeds_cat (encoder_hidden_states): {prompt_embeds_cat.shape}, dtype: {prompt_embeds_cat.dtype}")
        print(f"  pooled_prompt_embeds_cat (added_cond_kwargs['text_embeds']): {pooled_prompt_embeds_cat.shape}, dtype: {pooled_prompt_embeds_cat.dtype}")
        print(f"  time_ids_cat (added_cond_kwargs['time_ids']): {time_ids_cat.shape}, dtype: {time_ids_cat.dtype}")
        print(f"  timestep t: {t}")

        # UNet inference
        noise_pred_out = model.unet(
            latents_input, 
            t, 
            encoder_hidden_states=prompt_embeds_cat,
            added_cond_kwargs=added_cond_kwargs,
        )
        noise_pred = noise_pred_out["sample"] if isinstance(noise_pred_out, dict) else noise_pred_out


        # Perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        guided_noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # DDIM Inversion Step (reversed update rule)
        # prev_timestep = t - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
        # alpha_prod_t = model.scheduler.alphas_cumprod[t]
        # alpha_prod_t_prev = model.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod

        # Standard DDIM Inversion: x_t-1 = sqrt(alpha_t-1) * x0_pred + sqrt(1-alpha_t-1 - sigma_t^2) * eps_t + sigma_t * noise
        # Simplified for DDIM inversion (from PNDM/DDIM scheduler's add_noise logic essentially reversed)
        # x_0_pred = (latents - torch.sqrt(1 - alpha_prod_t) * guided_noise_pred) / torch.sqrt(alpha_prod_t)
        # latents = torch.sqrt(alpha_prod_t_prev) * x_0_pred + torch.sqrt(1 - alpha_prod_t_prev) * guided_noise_pred
        
        # Using the scheduler's step function for inversion might be complex as it's designed for forward.
        # Let's use the direct DDIM inversion formula if scheduler doesn't support reverse.
        # For now, assuming the provided step logic in original code was for forward DDIM,
        # and we need to adapt for inversion. The simplest way is to use the structure from diffusers inversion if available
        # or a known DDIM inversion implementation.
        # The existing "latents = model.scheduler.step(guided_noise_pred, t, latents)[\"prev_sample\"]" is FORWARD.
        # For DDIM inversion, the update is typically:
        # x_prev = sqrt(alpha_prev) * pred_x0 + sqrt(1 - alpha_prev) * noise_pred
        # where pred_x0 = (x_t - sqrt(1 - alpha_t) * noise_pred) / sqrt(alpha_t)

        alpha_prod_t = model.scheduler.alphas_cumprod[t]
        beta_prod_t = 1 - alpha_prod_t
        
        # Predict x0
        pred_x0 = (latents - beta_prod_t.sqrt() * guided_noise_pred) / alpha_prod_t.sqrt()
        
        # Get previous timestep
        idx = timesteps.tolist().index(t.item())
        if idx > 0: # We are not at the T-1 (first step of inversion)
            t_next_inversion = timesteps[idx-1] # This t_next_inversion > t
            alpha_bar_next_inversion = model.scheduler.alphas_cumprod[t_next_inversion]
            latents = torch.sqrt(alpha_bar_next_inversion) * pred_x0 + torch.sqrt(1 - alpha_bar_next_inversion) * guided_noise_pred
        else:
            pass
        latents = latents.to(dtype=dtype) # Ensure dtype after update
        all_latents.append(latents.clone())
    
    return latents, all_latents


def register_attention_control(model, controller):
    # 创建一个用于SDXL的强制投影层，从768投影到2048
    # force_projector is not used directly here anymore, projector created inside register_recr
    
    def ca_forward(self, place_in_unet): # 'self' is the Attention module instance
        def forward(
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            temb: Optional[torch.FloatTensor] = None,
            scale: float = 1.0,  # Added for SDXL
        ):
            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)

            batch_size, sequence_length, _ = (
                hidden_states.shape
                if encoder_hidden_states is None
                else encoder_hidden_states.shape
            )

            # 适配SDXL的维度
            sequence_length = hidden_states.shape[1]
            # inner_dim for query, key, value projections is derived from hidden_states.shape[2]
            # For SDXL, hidden_states dim can vary. to_q, to_k, to_v handle this.
            # The critical part is encoder_hidden_states' dimension for cross-attention.
            # dim = hidden_states.shape[2] # This dim is for hidden_states, not necessarily EHS

            if attention_mask is not None:
                attention_mask = self.prepare_attention_mask(
                    attention_mask, sequence_length, batch_size
                )
                attention_mask = attention_mask.view(
                    batch_size, self.heads, -1, attention_mask.shape[-1]
                )

            if self.group_norm is not None:
                hidden_states = self.group_norm(
                    hidden_states.transpose(1, 2)
                ).transpose(1, 2)

            query = self.to_q(hidden_states)

            is_cross = encoder_hidden_states is not None
            if not is_cross: 
                encoder_hidden_states = hidden_states
            else: # This is cross-attention
                # print(f"[DEBUG ca_forward] CROSS ATTENTION: EHS shape before projection: {encoder_hidden_states.shape if encoder_hidden_states is not None else 'None'}")
                if encoder_hidden_states is not None and hasattr(self, 'encoder_hid_proj') and self.encoder_hid_proj is not None:
                    if hasattr(self.to_k, 'in_features') and encoder_hidden_states.shape[-1] != self.to_k.in_features:
                        # Specific projection for 768-dim text embeddings to expected cross-attention dim (e.g., 2048)
                        if encoder_hidden_states.shape[-1] == 768 and self.to_k.in_features == (model.unet.config.cross_attention_dim or 2048):
                            try:
                                # print(f"[DEBUG ca_forward] Attempting projection. EHS dim: {encoder_hidden_states.shape[-1]}, to_k expects: {self.to_k.in_features}")
                                original_ehs_shape = encoder_hidden_states.shape
                                encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)
                                # print(f"[DEBUG ca_forward] Projection attempted. Original shape: {original_ehs_shape}, New EHS shape: {encoder_hidden_states.shape}")
                                if encoder_hidden_states.shape[-1] != self.to_k.in_features:
                                    print(f"[WARNING ca_forward] Projection did not result in expected dimension! Got {encoder_hidden_states.shape[-1]}, expected {self.to_k.in_features}")
                            except Exception as e:
                                print(f"[ERROR ca_forward] Projection failed: {e}. EHS shape: {encoder_hidden_states.shape}")
                        elif encoder_hidden_states.shape[-1] != self.to_k.in_features: # General mismatch
                             print(f"[WARNING ca_forward] EHS dim {encoder_hidden_states.shape[-1]} mismatches to_k in_features {self.to_k.in_features} but not the 768->{model.unet.config.cross_attention_dim or 2048} case.")
                    elif not hasattr(self.to_k, 'in_features'):
                        print("[WARNING ca_forward] self.to_k has no in_features attribute.")
                elif encoder_hidden_states is not None:
                     print(f"[WARNING ca_forward] Cross attention with EHS, but no encoder_hid_proj or EHS is None. EHS shape: {encoder_hidden_states.shape}")


            # Debug print just before to_k
            if encoder_hidden_states is not None:
                print(f"[DEBUG ca_forward] FINAL EHS shape before to_k: {encoder_hidden_states.shape}, self.to_k expects: {self.to_k.in_features if hasattr(self.to_k, 'in_features') else 'Unknown'}")
            else: # Should be self-attention if EHS became None (though logic above sets it to hidden_states for self-attn)
                print("[DEBUG ca_forward] EHS is effectively hidden_states (self-attention) or None before to_k")
            
            # For self-attention, encoder_hidden_states is hidden_states.
            # For cross-attention, encoder_hidden_states is the (potentially projected) context.
            # self.to_k expects input dim = self.to_k.in_features
            # For self-attention, this is hidden_states.shape[-1]
            # For cross-attention, this is model.unet.config.cross_attention_dim (e.g. 2048)
            
            # inner_dim for key/value is based on encoder_hidden_states' feature dim if it's cross-attn
            # or hidden_states' feature dim if self-attn.
            # The to_k, to_v layers are typically:
            # self-attn: Linear(hidden_dim, inner_dim)
            # cross-attn: Linear(cross_attention_dim, inner_dim)
            # So, encoder_hidden_states must match the respective in_features.
            
            inner_dim = self.to_k.out_features # Output dimension of K, Q, V projections
            head_dim = inner_dim // self.heads


            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            def reshape_heads_to_batch_dim(tensor):
                batch_size, seq_len, dim = tensor.shape
                head_size = self.heads
                tensor = tensor.reshape(
                    batch_size, seq_len, head_size, dim // head_size
                )
                tensor = tensor.permute(0, 2, 1, 3).reshape(
                    batch_size * head_size, seq_len, dim // head_size
                )
                return tensor

            query = reshape_heads_to_batch_dim(query)
            key = reshape_heads_to_batch_dim(key)
            value = reshape_heads_to_batch_dim(value)

            sim = torch.einsum("b i d, b j d -> b i j", query, key) * self.scale
            attn = sim.softmax(dim=-1)
            attn = controller(attn, is_cross, place_in_unet)
            out = torch.einsum("b i j, b j d -> b i d", attn, value)

            def reshape_batch_dim_to_heads(tensor):
                batch_size, seq_len, dim = tensor.shape
                head_size = self.heads
                tensor = tensor.reshape(
                    batch_size // head_size, head_size, seq_len, dim
                )
                tensor = tensor.permute(0, 2, 1, 3).reshape(
                    batch_size // head_size, seq_len, dim * head_size
                )
                return tensor

            out = reshape_batch_dim_to_heads(out)
            out = self.to_out[0](out)
            out = self.to_out[1](out)

            out = out / self.rescale_output_factor
            return out

        return forward

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == "Attention":
            # Check if this attention module needs a projector for 768-dim encoder_hidden_states
            # This usually applies to cross-attention layers in SDXL.
            # self.to_k.in_features would be model.unet.config.cross_attention_dim (e.g., 2048)
            # if it's a cross-attention layer. For self-attention, it's different.
            # We only add projector if it's likely for cross-attention taking 768-dim text embeds.
            expected_cross_attn_dim = getattr(model.unet.config, 'cross_attention_dim', 2048)
            if hasattr(net_.to_k, 'in_features') and net_.to_k.in_features == expected_cross_attn_dim:
                if not hasattr(net_, 'encoder_hid_proj') or net_.encoder_hid_proj is None:
                    # Input to projector is 768 (from text encoder)
                    # Output of projector must match net_.to_k.in_features (e.g., 2048)
                    projector = torch.nn.Linear(768, net_.to_k.in_features, bias=False).to(model.device, dtype=model.unet.dtype)
                    net_.encoder_hid_proj = projector
                    print(f"Attached encoder_hid_proj to Attention block in {place_in_unet}: Linear(768, {net_.to_k.in_features})")
            
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, "children"):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")
    controller.num_att_layers = cross_att_count


def reset_attention_control(model):
    # force_projector_reset is defined but not used if encoder_hid_proj is set by register_attention_control
    # and ca_forward_reset_def relies on it.
    # force_projector_reset = torch.nn.Linear(768, 2048).to(model.device).to(model.unet.dtype)
    
    def ca_forward_reset_def(self): # Renamed for clarity in this scope from 'ca_forward'
        def forward(
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            temb: Optional[torch.FloatTensor] = None,
            scale: float = 1.0,  # Added for SDXL
        ):
            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)

            batch_size, sequence_length, _ = (
                hidden_states.shape
                if encoder_hidden_states is None
                else encoder_hidden_states.shape
            )

            if attention_mask is not None:
                attention_mask = self.prepare_attention_mask(
                    attention_mask, sequence_length, batch_size
                )
                attention_mask = attention_mask.view(
                    batch_size, self.heads, -1, attention_mask.shape[-1]
                )

            if self.group_norm is not None:
                hidden_states = self.group_norm(
                    hidden_states.transpose(1, 2)
                ).transpose(1, 2)

            query = self.to_q(hidden_states)
            
            is_cross_reset = encoder_hidden_states is not None
            if not is_cross_reset:
                encoder_hidden_states = hidden_states
            # Original reset logic had self.norm_cross, which might not be standard for diffusers Attention
            # elif self.norm_cross: 
            #     encoder_hidden_states = self.norm_encoder_hidden_states(
            #         encoder_hidden_states
            #     )

            # Add projection logic to reset_attention_control as well
            if is_cross_reset and encoder_hidden_states is not None:
                print(f"[DEBUG ca_forward_reset] CROSS ATTENTION: EHS shape before projection: {encoder_hidden_states.shape}")
                if hasattr(self, 'encoder_hid_proj') and self.encoder_hid_proj is not None:
                    if hasattr(self.to_k, 'in_features') and encoder_hidden_states.shape[-1] != self.to_k.in_features:
                        # Specific projection for 768-dim text embeddings
                        if encoder_hidden_states.shape[-1] == 768 and self.to_k.in_features == (model.unet.config.cross_attention_dim or 2048):
                            try:
                                print(f"[DEBUG ca_forward_reset] Attempting projection. EHS dim: {encoder_hidden_states.shape[-1]}, to_k expects: {self.to_k.in_features}")
                                original_ehs_shape = encoder_hidden_states.shape
                                encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)
                                print(f"[DEBUG ca_forward_reset] Projection attempted. Original shape: {original_ehs_shape}, New EHS shape: {encoder_hidden_states.shape}")
                                if encoder_hidden_states.shape[-1] != self.to_k.in_features:
                                    print(f"[WARNING ca_forward_reset] Projection did not result in expected dimension! Got {encoder_hidden_states.shape[-1]}, expected {self.to_k.in_features}")
                            except Exception as e:
                                print(f"[ERROR ca_forward_reset] Projection failed: {e}. EHS shape: {encoder_hidden_states.shape}")
                        elif encoder_hidden_states.shape[-1] != self.to_k.in_features: # General mismatch
                             print(f"[WARNING ca_forward_reset] EHS dim {encoder_hidden_states.shape[-1]} mismatches to_k in_features {self.to_k.in_features} but not the 768->{model.unet.config.cross_attention_dim or 2048} case.")
                    elif not hasattr(self.to_k, 'in_features'):
                        print("[WARNING ca_forward_reset] self.to_k has no in_features attribute.")
                elif encoder_hidden_states is not None: # EHS exists but no projector
                     print(f"[WARNING ca_forward_reset] Cross attention, but no encoder_hid_proj. EHS shape: {encoder_hidden_states.shape}")
            
            if encoder_hidden_states is not None:
                print(f"[DEBUG ca_forward_reset] FINAL EHS shape before to_k: {encoder_hidden_states.shape}, self.to_k expects: {self.to_k.in_features if hasattr(self.to_k, 'in_features') else 'Unknown'}")
            else:
                print("[DEBUG ca_forward_reset] EHS is effectively hidden_states (self-attention) or None before to_k")

            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            def reshape_heads_to_batch_dim(tensor):
                batch_size, seq_len, dim = tensor.shape
                head_size = self.heads
                tensor = tensor.reshape(
                    batch_size, seq_len, head_size, dim // head_size
                )
                tensor = tensor.permute(0, 2, 1, 3).reshape(
                    batch_size * head_size, seq_len, dim // head_size
                )
                return tensor

            query = reshape_heads_to_batch_dim(query)
            key = reshape_heads_to_batch_dim(key)
            value = reshape_heads_to_batch_dim(value)

            sim = torch.einsum("b i d, b j d -> b i j", query, key) * self.scale

            attn = sim.softmax(dim=-1)
            out = torch.einsum("b i j, b j d -> b i d", attn, value)

            def reshape_batch_dim_to_heads(tensor):
                batch_size, seq_len, dim = tensor.shape
                head_size = self.heads
                tensor = tensor.reshape(
                    batch_size // head_size, head_size, seq_len, dim
                )
                tensor = tensor.permute(0, 2, 1, 3).reshape(
                    batch_size // head_size, seq_len, dim * head_size
                )
                return tensor

            out = reshape_batch_dim_to_heads(out)
            out = self.to_out[0](out)
            out = self.to_out[1](out)

            out = out / self.rescale_output_factor

            return out

        return forward

    def register_recr(net_):
        if net_.__class__.__name__ == "Attention":
            # Ensure the encoder_hid_proj is available if ca_forward_reset_def needs it.
            # It should have been set by register_attention_control.
            # If not, this reset might need to set it up too, or this ca_forward_reset_def needs to be robust.
            # For now, assume it's present from previous register_attention_control call.
            expected_cross_attn_dim = getattr(model.unet.config, 'cross_attention_dim', 2048)
            if hasattr(net_.to_k, 'in_features') and net_.to_k.in_features == expected_cross_attn_dim:
                if not hasattr(net_, 'encoder_hid_proj') or net_.encoder_hid_proj is None:
                     # Fallback: if projector wasn't set, or got removed, add it here too for reset to work with projection
                    projector = torch.nn.Linear(768, net_.to_k.in_features, bias=False).to(model.device, dtype=model.unet.dtype)
                    net_.encoder_hid_proj = projector
                    print(f"Attached encoder_hid_proj during reset_attention_control: Linear(768, {net_.to_k.in_features})")

            net_.forward = ca_forward_reset_def(net_) # Pass 'net_' as 'self' to the definition
        elif hasattr(net_, "children"):
            for net__ in net_.children():
                register_recr(net__)

    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            register_recr(net[1])
        elif "up" in net[0]:
            register_recr(net[1])
        elif "mid" in net[0]:
            register_recr(net[1])


def init_latent(latent, model, height, width, batch_size):
    # Ensure latents are in correct dtype
    latent = latent.to(dtype=model.unet.dtype, device=model.device)
    latents = latent.expand(batch_size, model.unet.config.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents


def diffusion_step(model, latents, context, t, guidance_scale, added_cond_kwargs):
    # Ensure data type consistency, using unet's dtype as the reference
    dtype = model.unet.dtype
    
    latents = latents.to(dtype=dtype)
    context = context.to(dtype=dtype)
    
    # Ensure all tensors in added_cond_kwargs are on the correct device and dtype
    processed_kwargs = {}
    for k, v in added_cond_kwargs.items():
        if isinstance(v, torch.Tensor):
            processed_kwargs[k] = v.to(device=model.device, dtype=dtype)
        else:
            processed_kwargs[k] = v # Non-tensor arguments
    
    # Prepare latents_input for CFG
    # latents is [actual_batch_size, C, H, W]
    # context is [2*actual_batch_size, seq, D]
    if latents.shape[0] * 2 == context.shape[0]:
        latents_input = torch.cat([latents] * 2, dim=0)
    else:
        # This case should ideally not be hit if inputs are prepared correctly for CFG
        latents_input = latents 
        if latents.shape[0] != context.shape[0]:
             print(f"[WARNING] diffusion_step: latents batch {latents.shape[0]} " \
                   f"does not match context batch {context.shape[0]} or context_batch/2. Passing latents as is.")


    print("\n[DIFFUSION_STEP Debug] Shapes for UNet input:")
    print(f"  latents_input: {latents_input.shape}, dtype: {latents_input.dtype}")
    print(f"  context (encoder_hidden_states): {context.shape}, dtype: {context.dtype}")
    if "text_embeds" in processed_kwargs:
        print(f"  processed_kwargs['text_embeds']: {processed_kwargs['text_embeds'].shape}, dtype: {processed_kwargs['text_embeds'].dtype}")
    if "time_ids" in processed_kwargs:
        print(f"  processed_kwargs['time_ids']: {processed_kwargs['time_ids'].shape}, dtype: {processed_kwargs['time_ids'].dtype}")
    print(f"  timestep t: {t}")
    
    noise_pred_out = model.unet(
        latents_input, 
        t, 
        encoder_hidden_states=context,
        added_cond_kwargs=processed_kwargs,
    )
    noise_pred = noise_pred_out["sample"] if isinstance(noise_pred_out, dict) else noise_pred_out
    
    noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    return latents


def latent2image(vae, latents):
    # Ensure latents are in correct dtype for VAE
    latents = latents.to(dtype=vae.dtype)
    latents = 1 / 0.13025 * latents  # SDXL scaling factor
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()  # Ensure float32 for numpy
    image = (image * 255).astype(np.uint8)
    return image


@torch.enable_grad()
def diffattack(
        model,
        label,
        controller,
        num_inference_steps: int = 20,
        guidance_scale: float = 2.5,
        image=None,
        model_name="inception",
        save_path=r"C:\Users\PC\Desktop\output",
        res=224,
        start_step=15,
        iterations=30,
        verbose=True,
        topN=1,
        args=None
):
    # Get model's dtype for consistent tensor operations
    model_dtype = model.unet.dtype
    
    if args.dataset_name == "imagenet_compatible":
        from dataset_caption import imagenet_label
    elif args.dataset_name == "cub_200_2011":
        from dataset_caption import CUB_label as imagenet_label
    elif args.dataset_name == "standford_car":
        from dataset_caption import stanfordCar_label as imagenet_label
    else:
        raise NotImplementedError

    label = torch.from_numpy(label).long().cuda()

    model.vae.requires_grad_(False)
    # For SDXL, disable gradient for both text encoders
    model.text_encoder.requires_grad_(False)
    model.text_encoder_2.requires_grad_(False)
    model.unet.requires_grad_(False)

    classifier = other_attacks.model_selection(model_name).eval()
    classifier.requires_grad_(False)

    height = width = res

    test_image = image.resize((height, height), resample=Image.LANCZOS)
    test_image = np.float32(test_image) / 255.0
    test_image = test_image[:, :, :3]
    test_image[:, :, ] -= (np.float32(0.485), np.float32(0.456), np.float32(0.406))
    test_image[:, :, ] /= (np.float32(0.229), np.float32(0.224), np.float32(0.225))
    test_image = test_image.transpose((2, 0, 1))
    test_image = torch.from_numpy(test_image).unsqueeze(0)

    pred = classifier(test_image.cuda())
    pred_accuracy_clean = (torch.argmax(pred, 1).detach() == label).sum().item() / len(label)
    print("\nAccuracy on benign examples: {}%".format(pred_accuracy_clean * 100))

    logit = torch.nn.Softmax()(pred)
    print("gt_label:", label[0].item(), "pred_label:", torch.argmax(pred, 1).detach().item(), "pred_clean_logit",
          logit[0, label[0]].item())

    _, pred_labels = pred.topk(topN, largest=True, sorted=True)

    target_prompt = " ".join([imagenet_label.refined_Label[label.item()] for i in range(1, topN)])
    prompt = [imagenet_label.refined_Label[label.item()] + " " + target_prompt] * 2
    print("prompt generate: ", prompt[0], "\tlabels: ", pred_labels.cpu().numpy().tolist())

    # For SDXL, we need to handle tokenization differently
    # Get tokenized IDs for the first encoder
    # This part needs to be adapted for SDXL tokenizers (tokenizer and tokenizer_2)
    # true_label_tokens = model.tokenizer.encode(imagenet_label.refined_Label[label.item()])
    # target_label_tokens = model.tokenizer.encode(target_prompt)
    # print("decoder: ", true_label_tokens, target_label_tokens)
    
    # For SDXL, use both tokenizers if needed for specific attention map interpretation
    # For now, let's assume model.tokenizer (the first one) is primary for word-to-token mapping visibility
    true_label_words = imagenet_label.refined_Label[label.item()].split()
    true_label_token_ids = [model.tokenizer.encode(word, add_special_tokens=False) for word in true_label_words]
    
    target_prompt_words = target_prompt.split()
    target_label_token_ids = [model.tokenizer.encode(word, add_special_tokens=False) for word in target_prompt_words]

    print(f"True label token IDs (word-wise): {true_label_token_ids}")
    print(f"Target label token IDs (word-wise): {target_label_token_ids}")
    
    # The attention map slicing [:, :, 1: len(true_label) - 1] assumes BOS/EOS tokens.
    # Need to adjust if using tokenizer that doesn't add them by default or if we encoded word by word.
    # For simplicity, let's get the full token sequence from the main tokenizer for the true label part of the prompt.
    # The prompt used for attention aggregation is the full attack prompt.
    # attack_prompt_for_attn = prompt[0] # This is the cond_prompt + target_prompt from diffattack
    # tokenized_attack_prompt = model.tokenizer(attack_prompt_for_attn, padding="max_length", max_length=model.tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids[0]
    
    # Find start and end index of the true_label part in the full attack prompt's tokens
    # This is complex if target_prompt is involved. The original code had:
    # before_true_label_attention_map = before_attention_map[:, :, 1: len(true_label_tokens) - 1]
    # This implies `true_label_tokens` are the first part of the prompt.
    # The prompt for attention is `prompt` which is `[cond_str_for_attack, cond_str_for_attack]`
    # aggregate_attention takes `prompt` (list of 2)
    # And select=0 or select=1.
    # Let's get tokens for `prompt[0]`
    
    # Tokens for the actual prompt string used in the attention maps
    # The `prompt` variable in `diffattack` (used for `aggregate_attention`) is `[cond_str]*2`.
    # We are interested in the tokens of `cond_str`.
    # `cond_str = imagenet_label.refined_Label[label.item()] + " " + target_prompt`
    
    # Get tokens for the "true label" part only: imagenet_label.refined_Label[label.item()]
    true_label_only_str = imagenet_label.refined_Label[label.item()]
    tokens_for_true_label_part = model.tokenizer.encode(true_label_only_str) # includes BOS/EOS
    num_true_label_tokens_inc_special = len(tokens_for_true_label_part)
    
    # The slicing `1:len(true_label_tokens)-1` removes BOS and EOS.
    # So, effectively `num_true_label_tokens_exc_special = num_true_label_tokens_inc_special - 2`.
    # The attention map indices should correspond to these tokens.
    # The aggregate_attention function uses the full prompt list.
    # The attention map's last dimension corresponds to tokens of prompt[select].

    # DDIM Inversion: Use a single conditional prompt string
    single_cond_prompt = prompt[0] # Assuming prompt was [cond_str]*2 initially
    # Or more robustly:
    # if isinstance(prompt, list) and len(prompt) > 0:
    #    single_cond_prompt = prompt[0]
    # else: # Should not happen based on current diffattack logic
    #    single_cond_prompt = prompt if isinstance(prompt, str) else " "
    
    print(f"Using single_cond_prompt for DDIM Inversion: \'{single_cond_prompt}\'")

    # Ensure controller is reset and attention control is registered before ddim_reverse_sample
    controller.reset()
    register_attention_control(model, controller)

    latent, inversion_latents = ddim_reverse_sample(image, single_cond_prompt, model,
                                                    num_inference_steps,
                                                    0, res=height) # guidance_scale = 0 for inversion

    init_prompt = [prompt[0]]
    batch_size = len(init_prompt) # This will be 1
    latent = inversion_latents[start_step - 1]

    """
            ===============================================================================
            === Good initial reconstruction by optimizing the unconditional embeddings ====
            ======================= Details please refer to Section 3.4 ===================
            ===============================================================================
    """
    # Use encode_prompt with the actual string from init_prompt
    init_prompt_string = init_prompt[0]
    init_prompt_data = encode_prompt(model, init_prompt_string)
    
    # For null prompt, also pass the string ""
    null_prompt_string = "" # batch_size is 1, so effectively [""][0]
    null_data = encode_prompt(model, null_prompt_string)
    
    # trainable_uncond_embeddings are the embeddings for "" (empty string)
    # null_data["prompt_embeds"] contains the conditional embedding for null_prompt_string ("")
    uncond_embeddings = null_data["prompt_embeds"].clone().to(dtype=model_dtype)
    uncond_embeddings.requires_grad_(True)
    
    # text_embeddings are the conditional embeddings for init_prompt_string
    text_embeddings = init_prompt_data["prompt_embeds"].clone().to(dtype=model_dtype)
    
    # Pooled embeddings
    # uncond_pooled_embeds are for null_prompt_string ("")
    uncond_pooled_embeds = null_data["pooled_prompt_embeds"].clone().to(dtype=model_dtype)
    # text_pooled_embeds are for init_prompt_string
    text_pooled_embeds = init_prompt_data["pooled_prompt_embeds"].clone().to(dtype=model_dtype)

    all_uncond_emb = []
    all_uncond_pooled_emb = []  # 存储优化后的pooled embeddings
    latent = latent.to(dtype=model_dtype)
    latent, latents = init_latent(latent, model, height, width, batch_size)
    # Clamp latent to valid range
    latent = torch.clamp(latent, -1.0, 1.0)
    latents = torch.clamp(latents, -1.0, 1.0)

    optimizer = optim.AdamW([uncond_embeddings], lr=1e-2)
    loss_func = torch.nn.MSELoss()

    # 准备SDXL的time_ids
    original_size = (res, res)
    crop_coords = (0, 0)
    target_size = (res, res)
    time_ids = torch.tensor([*original_size, *crop_coords, *target_size], 
                          dtype=model_dtype, device=model.device).unsqueeze(0).repeat(2*batch_size, 1)

    # The DDIM should begin from 1, as the inversion cannot access X_T but only X_{T-1}
    for ind, t in enumerate(tqdm(model.scheduler.timesteps[1 + start_step - 1:], desc="Optimize_uncond_embed")):
        for _ in range(10 + 2 * ind):
            # 组合当前embeddings
            prompt_embeds = torch.cat([uncond_embeddings, text_embeddings], dim=0)  # [2, 77, 768]
            pooled_prompt_embeds = torch.cat([uncond_pooled_embeds, text_pooled_embeds], dim=0)  # [2, 1280]
            
            # SDXL的added_cond_kwargs
            added_cond_kwargs = {
                "text_embeds": pooled_prompt_embeds,  # [2, 1280]
                "time_ids": time_ids[:2*batch_size]  # [2, 6]
            }
            
            out_latents = diffusion_step(model, latents, prompt_embeds, t, guidance_scale, added_cond_kwargs)
            optimizer.zero_grad()
            target_latents = inversion_latents[start_step - 1 + ind + 1].to(dtype=model_dtype)
            loss = loss_func(out_latents, target_latents)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            # 使用更新后的uncond_embeddings
            prompt_embeds = torch.cat([uncond_embeddings, text_embeddings], dim=0)  # [2, 77, 768]
            pooled_prompt_embeds = torch.cat([uncond_pooled_embeds, text_pooled_embeds], dim=0)  # [2, 1280]
            
            added_cond_kwargs = {
                "text_embeds": pooled_prompt_embeds,  # [2, 1280]
                "time_ids": time_ids[:2*batch_size]
            }
            
            latents = diffusion_step(model, latents, prompt_embeds, t, guidance_scale, added_cond_kwargs).detach()
            all_uncond_emb.append(uncond_embeddings.detach().clone())
            all_uncond_pooled_emb.append(uncond_pooled_embeds.detach().clone())

    """
            ==========================================
            ============ Latents Attack ==============
            ==== Details please refer to Section 3 ===
            ==========================================
    """
    
    uncond_embeddings.requires_grad_(False)

    register_attention_control(model, controller)

    batch_size = len(prompt)
    
    # 使用encode_prompt处理攻击prompt
    # prompt is [cond_str, cond_str]. We need embeddings for cond_str.
    # encode_prompt takes a single string.
    attack_prompt_string = prompt[0] # Since prompt[0] and prompt[1] are the same
    attack_prompt_data = encode_prompt(model, attack_prompt_string)

    # These are the conditional embeddings for the attack prompt string
    cond_embeddings_for_attack_prompt = attack_prompt_data["prompt_embeds"]       # Shape [1, 77, D]
    cond_pooled_for_attack_prompt = attack_prompt_data["pooled_prompt_embeds"] # Shape [1, 1280]
    
    #攻击阶段contexts/pooled_contexts构造修正
    # contexts[i] should be [uncond_orig, cond_orig, uncond_adv, cond_adv] -> [4, 77, D]
    # pooled_contexts[i] should be [uncond_pooled_orig, cond_pooled_orig, uncond_pooled_adv, cond_pooled_adv] -> [4, 1280]
    contexts = []
    pooled_contexts = []
    for i in range(len(all_uncond_emb)):
        # optimized_uncond_emb is from the optimization loop, for the "original" path
        optimized_uncond_emb = all_uncond_emb[i]               # Shape [1, 77, D]
        optimized_uncond_pooled = all_uncond_pooled_emb[i]   # Shape [1, 1280]

        # Conditional embeddings for both original and adversarial paths are based on text_embeddings
        # (derived from init_prompt[0], which is same content as attack_prompt_string)
        
        # For UNet's encoder_hidden_states:
        # Order: [uncond_original, cond_original, uncond_adversarial, cond_adversarial]
        # We use optimized_uncond_emb for both uncond_original and uncond_adversarial here,
        # as the controller will introduce differences.
        # text_embeddings (from init_prompt[0]) serves as cond_original and cond_adversarial.
        current_context = torch.cat([
            optimized_uncond_emb,    # Uncond for Original item
            text_embeddings,         # Cond for Original item (from init_prompt[0])
            optimized_uncond_emb,    # Uncond for Adversarial item
            text_embeddings          # Cond for Adversarial item (from init_prompt[0])
        ], dim=0)  # Expected shape [4, 77, D]
        contexts.append(current_context)
        
        # For UNet's added_cond_kwargs["text_embeds"]:
        current_pooled = torch.cat([
            optimized_uncond_pooled, # Uncond Pooled for Original
            text_pooled_embeds,      # Cond Pooled for Original (from init_prompt[0])
            optimized_uncond_pooled, # Uncond Pooled for Adversarial
            text_pooled_embeds       # Cond Pooled for Adversarial (from init_prompt[0])
        ], dim=0)  # Expected shape [4, 1280]
        pooled_contexts.append(current_pooled)

    # 准备SDXL条件参数
    # batch_size for attack is 2 (original and adversarial paths)
    # time_ids_list needs to be [2 * actual_batch_size_for_unet, 6] = [4, 6]
    # current batch_size = len(prompt) is 2. So repeat(2 * 2, 1) = repeat(4,1). This is correct.
    time_ids_list = torch.tensor([*original_size, *crop_coords, *target_size], 
                              device=model.device, 
                              dtype=model_dtype).unsqueeze(0).repeat(batch_size * 2, 1)
    
    added_cond_kwargs_list = []
    for i in range(len(contexts)):
        added_cond_kwargs_list.append({
            "text_embeds": pooled_contexts[i],  # [2, 1280]
            "time_ids": time_ids_list
        })

    original_latent = latent.clone()

    latent.requires_grad_(True)

    optimizer = optim.AdamW([latent], lr=1e-3)
    cross_entro = torch.nn.CrossEntropyLoss()
    init_image = preprocess(image, res).to(dtype=model_dtype)

    #  "Pseudo" Mask for better Imperceptibility, yet sacrifice the transferability. Details please refer to Appendix D.
    apply_mask = args.is_apply_mask
    hard_mask = args.is_hard_mask
    if apply_mask:
        init_mask = None
    else:
        init_mask = torch.ones([1, 1, *init_image.shape[-2:]]).cuda()

    pbar = tqdm(range(iterations), desc="Iterations")
    for _, _ in enumerate(pbar):
        controller.loss = 0

        #  The DDIM should begin from 1, as the inversion cannot access X_T but only X_{T-1}
        controller.reset()
        # For CFG, latents_input should be [2*actual_batch_size_for_attack, C, H, W]
        # Here, original_latent is [1,C,H,W], latent (adv) is [1,C,H,W]
        # So torch.cat([original_latent, latent]) is [2,C,H,W]. This is correct for the 2 conceptual images.
        latents_for_diffusion_step = torch.cat([original_latent, latent]) # This is correct, batch_size=2 for attack

        for ind, t in enumerate(model.scheduler.timesteps[1 + start_step - 1:]):
            # current_context and current_cond_kwargs are for a batch of 2 (original and adv)
            # Each of these 2 needs its own CFG.
            # contexts[ind] has shape [2*2, 77, 768] (uncond for orig, cond for orig, uncond for adv, cond for adv)
            # current_cond_kwargs["text_embeds"] has shape [2*2, 1280]
            # latents_for_diffusion_step is [2, C, H, W]
            # diffusion_step expects latents [B_actual,...] and context [2*B_actual,...]
            # So, we need to call diffusion_step for original_latent and then for latent (adv) separately if we keep diffusion_step simple,
            # OR modify diffusion_step to handle a batch of 2 actual images, each with CFG.

            # Simpler: if controller.batch_size is 2, it implies we process 2 items.
            # The current_context already has embeddings for 2 items, each with its CFG pair.
            # So current_context is shape [4, 77, 768]
            # current_cond_kwargs["text_embeds"] is shape [4, 1280]
            # latents_for_diffusion_step is [2, C, H, W]
            # We need to expand latents_for_diffusion_step for CFG for EACH of the 2 items.
            # This means latents_for_diffusion_step should become [4, C, H, W] for the unet.
            # But diffusion_step's current implementation is:
            # latents_input = torch.cat([latents] * 2) if latents.shape[0] == context.shape[0] // 2 else latents
            # If latents is [2,C,H,W] and context is [4, ...], then context.shape[0]//2 = 2.
            # So latents_input = torch.cat([latents]*2) = [4,C,H,W]. This is correct!
            
            # Ensure context and added_cond_kwargs are correctly indexed for this step
            # contexts and added_cond_kwargs_list are indexed by `ind` of scheduler timesteps
            # These already contain the CFG-doubled embeddings for the 2 conceptual images.
            
            # The `diffusion_step` expects:
            #   latents: [actual_batch_size, C, H, W] -> here [2, C, H, W]
            #   context: [2*actual_batch_size, seq, D] -> here contexts[ind] is [4, seq, D]
            #   added_cond_kwargs["text_embeds"]: [2*actual_batch_size, D_pooled] -> here current_cond_kwargs["text_embeds"] is [4, D_pooled]
            #   added_cond_kwargs["time_ids"]: [2*actual_batch_size, 6] -> here current_cond_kwargs["time_ids"] is [4, 6] (from earlier repeat with batch_size*2)
            
            # Check `time_ids_list` creation for attack phase:
            # time_ids_list = torch.tensor(...).unsqueeze(0).repeat(batch_size * 2, 1) where batch_size = len(prompt) = 2. So repeat(4,1) -> [4,6]. Correct.

            latents_for_diffusion_step = diffusion_step(model, latents_for_diffusion_step, contexts[ind], t, guidance_scale, added_cond_kwargs_list[ind])
            # Output latents_for_diffusion_step from diffusion_step is [actual_batch_size, C,H,W] -> [2, C,H,W]

        # ... after loop ...
        # The attention map slicing:
        # before_true_label_attention_map = before_attention_map[:, :, 1: len(true_label_tokens) - 1]
        # Let's adjust `true_label_tokens` to be consistent.
        # `true_label` is the numeric label. `imagenet_label.refined_Label[label.item()]` is the string.
        # For `aggregate_attention`, the `prompts` arg is `diffattack`'s `prompt` variable, which is `[cond_str, cond_str]`.
        # `controller` has stored attentions for these. `select=0` takes first, `select=1` takes second.
        
        # For slicing, we need tokenization of the "true label" part of `prompt[0]` or `prompt[1]`.
        # Example: prompt[0] = "A photo of a cat playing with yarn"
        # true_label part might be "cat".
        # The `1: len(true_label_tokens)-1` was based on `true_label = model.tokenizer.encode(imagenet_label.refined_Label[label.item()])`
        # Let's use this encoding for consistency for slicing.
        true_label_str_for_slicing = imagenet_label.refined_Label[label.item()]
        true_label_tokens_for_slicing = model.tokenizer.encode(true_label_str_for_slicing) # Includes BOS/EOS usually

        # Attention map shape is [H, W, num_tokens_in_prompt[select]]
        # If true_label_tokens_for_slicing includes BOS/EOS, then slicing 1:-1 removes them.
        # The number of actual word tokens is len(true_label_tokens_for_slicing) - 2.
        # The slice should be:
        start_idx_attn = 1 # After BOS
        end_idx_attn = len(true_label_tokens_for_slicing) - 1 # Before EOS
        
        # Compute attention maps for before and after
        before_attention_map = aggregate_attention(prompt, controller, args.res // 32, ("up", "down"), True, 0, is_cpu=False)
        after_attention_map = aggregate_attention(prompt, controller, args.res // 32, ("up", "down"), True, 1, is_cpu=False)
        before_true_label_attention_map = before_attention_map[:, :, start_idx_attn:end_idx_attn]
        after_true_label_attention_map = after_attention_map[:, :, start_idx_attn:end_idx_attn]

        if init_mask is None:
            # Convert attention map to float32 for image processing operations
            att_map = before_true_label_attention_map.detach().clone().float().mean(-1)
            att_map_max = att_map.max()
            att_map = att_map / (att_map_max + 1e-8)
            att_map = torch.nan_to_num(att_map, nan=0.0, posinf=0.0, neginf=0.0)
            init_mask = torch.nn.functional.interpolate(
                att_map.unsqueeze(0).unsqueeze(0),
                init_image.shape[-2:], 
                mode="bilinear"
            ).clamp(0, 1)
            
            if hard_mask:
                init_mask = init_mask.gt(0.5).float()
                
        # Ensure all tensors have the right dtype before operations
        init_mask = init_mask.to(dtype=model.vae.dtype)
        init_image = init_image.to(dtype=model.vae.dtype)
        
        # Decode VAE latents
        latents_for_decode = latents_for_diffusion_step.to(dtype=model.vae.dtype)
        init_out_image = model.vae.decode(1 / 0.13025 * latents_for_decode)['sample'][1:] * init_mask + (1 - init_mask) * init_image

        # Check for nan in decoded image
        if torch.isnan(init_out_image).any():
            print("init_out_image contains nan")

        out_image = (init_out_image / 2 + 0.5).clamp(0, 1)
        out_image = out_image.permute(0, 2, 3, 1)
        mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=out_image.dtype, device=out_image.device)
        std = torch.as_tensor([0.229, 0.224, 0.225], dtype=out_image.dtype, device=out_image.device)
        out_image = out_image[:, :, :].sub(mean).div(std)
        out_image = out_image.permute(0, 3, 1, 2)

        # Convert to float32 for classifier
        out_image = out_image.float()
        
        # For datasets like CUB, Standford Car, the logit should be divided by 10, or there will be gradient Vanishing.
        if args.dataset_name != "imagenet_compatible":
            pred = classifier(out_image) / 10
        else:
            pred = classifier(out_image)
        
        pred_label = torch.argmax(pred, 1).detach()
        pred_accuracy = (torch.argmax(pred, 1).detach() == label).sum().item() / len(label)
        print("Accuracy on adversarial examples: {}%".format(pred_accuracy * 100))

        attack_loss = - cross_entro(pred, label) * args.attack_loss_weight
        if torch.isnan(attack_loss):
            print("attack_loss is nan")

        # "Deceive" Strong Diffusion Model. Details please refer to Section 3.3
        variance_cross_attn_loss = after_true_label_attention_map.var() * args.cross_attn_loss_weight
        if torch.isnan(variance_cross_attn_loss):
            print("variance_cross_attn_loss is nan")

        # Preserve Content Structure. Details please refer to Section 3.4
        self_attn_loss = controller.loss * args.self_attn_loss_weight
        if torch.isnan(self_attn_loss):
            print("self_attn_loss is nan")

        loss = self_attn_loss + attack_loss + variance_cross_attn_loss
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=-1e6)
        if torch.isnan(loss):
            print("loss is nan before backward")
        assert not torch.isnan(loss), "loss is nan before backward"

        if verbose:
            pbar.set_postfix_str(
                f"attack_loss: {attack_loss.item():.5f} "
                f"variance_cross_attn_loss: {variance_cross_attn_loss.item():.5f} "
                f"self_attn_loss: {self_attn_loss.item():.5f} "
                f"loss: {loss.item():.5f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        controller.loss = 0
        controller.reset()

        latents = torch.cat([original_latent, latent])

        for ind, t in enumerate(model.scheduler.timesteps[1 + start_step - 1:]):
            # 确保使用正确的上下文和条件参数
            current_context = contexts[ind]
            current_cond_kwargs = added_cond_kwargs_list[ind]
            latents = diffusion_step(model, latents, current_context, t, guidance_scale, current_cond_kwargs)

    # VAE decode with correct dtype
    latents = latents.to(dtype=model.vae.dtype)
    out_images = model.vae.decode(1 / 0.13025 * latents.detach())['sample']
    
    # Ensure consistent dtype operations
    init_mask = init_mask.to(dtype=out_images.dtype)
    init_image = init_image.to(dtype=out_images.dtype)
    
    # Get the adversarial image (index 1)
    out_image = out_images[1:] * init_mask + (1 - init_mask) * init_image
    
    # Convert to format needed for classifier
    out_image = (out_image / 2 + 0.5).clamp(0, 1)
    out_image = out_image.permute(0, 2, 3, 1)
    mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=out_image.dtype, device=out_image.device)
    std = torch.as_tensor([0.229, 0.224, 0.225], dtype=out_image.dtype, device=out_image.device)
    out_image = out_image[:, :, :].sub(mean).div(std)
    out_image = out_image.permute(0, 3, 1, 2)

    # Convert to float32 for classifier
    out_image = out_image.float()
    
    # For datasets like CUB, Standford Car, the logit should be divided by 10, or there will be gradient Vanishing.
    if args.dataset_name != "imagenet_compatible":
        pred = classifier(out_image) / 10
    else:
        pred = classifier(out_image)
        
    pred_label = torch.argmax(pred, 1).detach()
    pred_accuracy = (torch.argmax(pred, 1).detach() == label).sum().item() / len(label)
    print("Accuracy on adversarial examples: {}%".format(pred_accuracy * 100))

    logit = torch.nn.Softmax()(pred)
    print("after_pred:", pred_label, logit[0, pred_label[0]])
    print("after_true:", label, logit[0, label[0]])

    """
            ==========================================
            ============= Visualization ==============
            ==========================================
    """

    image = latent2image(model.vae, latents.detach())

    real = (init_image / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    perturbed = image[1:].astype(np.float32) / 255 * init_mask.squeeze().unsqueeze(-1).cpu().numpy() + (
            1 - init_mask.squeeze().unsqueeze(-1).cpu().numpy()) * real
    image = (perturbed * 255).astype(np.uint8)
    view_images(np.concatenate([real, perturbed]) * 255, show=False,
                save_path=save_path + "_diff_{}_image_{}.png".format(model_name,
                                                                     "ATKSuccess" if pred_accuracy == 0 else "Fail"))
    view_images(perturbed * 255, show=False, save_path=save_path + "_adv_image.png")

    L1 = LpDistance(1)
    L2 = LpDistance(2)
    Linf = LpDistance(float("inf"))

    print("L1: {}\tL2: {}\tLinf: {}".format(L1(real, perturbed), L2(real, perturbed), Linf(real, perturbed)))

    diff = perturbed - real
    diff = (diff - diff.min()) / (diff.max() - diff.min()) * 255

    view_images(diff.clip(0, 255), show=False,
                save_path=save_path + "_diff_relative.png")

    diff = (np.abs(perturbed - real) * 255).astype(np.uint8)
    view_images(diff.clip(0, 255), show=False,
                save_path=save_path + "_diff_absolute.png")

    reset_attention_control(model)

    # utils.show_cross_attention(prompt, model.tokenizer, controller, res=args.res // 32, from_where=("up", "down"),
    #                            save_path=r"{}_crossAttentionBefore.jpg".format(save_path))
    # utils.show_cross_attention(prompt, model.tokenizer, controller, res=args.res // 32, from_where=("up", "down"),
    #                            save_path=r"{}_crossAttentionAfter.jpg".format(save_path), select=1)
    # utils.show_self_attention_comp(prompt, controller, res=14, from_where=("up", "down"),
    #                                save_path=r"{}_selfAttentionBefore.jpg".format(save_path))
    # utils.show_self_attention_comp(prompt, controller, res=14, from_where=("up", "down"),
    #                                save_path=r"{}_selfAttentionAfter.jpg".format(save_path), select=1)

    return image[0], 0, 0
