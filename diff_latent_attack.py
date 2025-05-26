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
def encode_prompt(model, prompt):
    """为SDXL编码提示词，返回用于UNet的完整嵌入张量字典"""
    # text_encoder: CLIPTextModel, 输出 [batch, 77, 768]
    text_inputs = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(model.device)
    cond_prompt_embeds = model.text_encoder(text_inputs.input_ids)[0]  # [batch, 77, 768]

    # text_encoder_2: CLIPTextModelWithProjection, 输出 [batch, 1280]（直接pooled）
    add_text_inputs = model.tokenizer_2(
        prompt,
        padding="max_length",
        max_length=model.tokenizer_2.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(model.device)
    cond_pooled_prompt_embeds = model.text_encoder_2(add_text_inputs.input_ids)[0]  # [batch, 1280]

    # uncond
    uncond_input = model.tokenizer(
        [""] * len(prompt),
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(model.device)
    uncond_prompt_embeds = model.text_encoder(uncond_input.input_ids)[0]  # [batch, 77, 768]

    uncond_add_input = model.tokenizer_2(
        [""] * len(prompt),
        padding="max_length",
        max_length=model.tokenizer_2.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(model.device)
    uncond_pooled_prompt_embeds = model.text_encoder_2(uncond_add_input.input_ids)[0]  # [batch, 1280]

    print("[ENCODE] cond_prompt_embeds shape:", cond_prompt_embeds.shape)
    print("[ENCODE] cond_pooled_prompt_embeds shape:", cond_pooled_prompt_embeds.shape)
    print("[ENCODE] uncond_prompt_embeds shape:", uncond_prompt_embeds.shape)
    print("[ENCODE] uncond_pooled_prompt_embeds shape:", uncond_pooled_prompt_embeds.shape)
    return {
        "prompt_embeds": cond_prompt_embeds,
        "pooled_prompt_embeds": cond_pooled_prompt_embeds,
        "uncond_prompt_embeds": uncond_prompt_embeds,
        "uncond_pooled_prompt_embeds": uncond_pooled_prompt_embeds
    }


@torch.no_grad()
def ddim_reverse_sample(image, prompt, model, num_inference_steps: int = 20, guidance_scale: float = 2.5,
                       res=512):
    """
            ==========================================
            ============ DDIM Inversion ==============
            ==========================================
    """
    batch_size = 1

    # 使用我们自定义的encode_prompt方法
    prompt_data = encode_prompt(model, prompt)
    prompt_embeds = prompt_data["prompt_embeds"]  # [batch, 77, 768]
    pooled_prompt_embeds = prompt_data["pooled_prompt_embeds"]  # [batch, 1280]
    uncond_prompt_embeds = prompt_data["uncond_prompt_embeds"]  # [batch, 77, 768]
    uncond_pooled_prompt_embeds = prompt_data["uncond_pooled_prompt_embeds"]  # [batch, 1280]
    
    # Initialize the reverse process
    model.scheduler.set_timesteps(num_inference_steps)

    latents = encoder(image, model, res=res)
    timesteps = model.scheduler.timesteps.flip(0)
    all_latents = [latents]
    original_size = (res, res)
    crop_coords = (0, 0)
    target_size = (res, res)
    time_ids = torch.tensor([*original_size, *crop_coords, *target_size], device=model.device).unsqueeze(0).repeat(batch_size, 1)
    uncond_time_ids = time_ids.clone()
    for t in tqdm(timesteps[:-1], desc="DDIM_inverse"):
        # 每步都重新拼接CFG输入，确保batch一致
        latents_input = torch.cat([latents, latents], dim=0)  # [2*batch, ...]
        prompt_embeds_cat = torch.cat([uncond_prompt_embeds, prompt_embeds], dim=0)  # [2*batch, 77, 768]
        pooled_prompt_embeds_cat = torch.cat([uncond_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)  # [2*batch, 1280]
        time_ids_cat = torch.cat([uncond_time_ids, time_ids], dim=0)  # [2*batch, 6]
        added_cond_kwargs = {
            "text_embeds": pooled_prompt_embeds_cat,  # [2*batch, 1280]
            "time_ids": time_ids_cat
        }
        print("[DDIM] prompt_embeds shape:", prompt_embeds_cat.shape)
        print("[DDIM] pooled_prompt_embeds shape:", pooled_prompt_embeds_cat.shape)
        print("[DDIM] latents_input shape:", latents_input.shape)
        print("[DDIM] added_cond_kwargs['text_embeds'] shape:", added_cond_kwargs['text_embeds'].shape)
        print("[DDIM] added_cond_kwargs['time_ids'] shape:", added_cond_kwargs['time_ids'].shape)
        noise_pred = model.unet(
            latents_input, 
            t, 
            encoder_hidden_states=prompt_embeds_cat,
            added_cond_kwargs=added_cond_kwargs
        )["sample"]
        # 拆分uncond/cond
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
        all_latents.append(latents)

    #  all_latents[N] -> N: DDIM steps  (X_{T-1} ~ X_0)
    return latents, all_latents


def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
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
                # scaled_dot_product_attention expects attention_mask shape to be
                # (batch, heads, source_length, target_length)
                attention_mask = attention_mask.view(
                    batch_size, self.heads, -1, attention_mask.shape[-1]
                )  # type: ignore

            if self.group_norm is not None:
                hidden_states = self.group_norm(
                    hidden_states.transpose(1, 2)
                ).transpose(1, 2)

            query = self.to_q(hidden_states)

            is_cross = encoder_hidden_states is not None
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(
                    encoder_hidden_states
                )
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
    def ca_forward(self):
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
                # scaled_dot_product_attention expects attention_mask shape to be
                # (batch, heads, source_length, target_length)
                attention_mask = attention_mask.view(
                    batch_size, self.heads, -1, attention_mask.shape[-1]
                )  # type: ignore

            if self.group_norm is not None:
                hidden_states = self.group_norm(
                    hidden_states.transpose(1, 2)
                ).transpose(1, 2)

            query = self.to_q(hidden_states)
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(
                    encoder_hidden_states
                )

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
            net_.forward = ca_forward(net_)
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


def diffusion_step(model, latents, context, t, guidance_scale, added_cond_kwargs=None):
    """对给定的潜变量执行一步扩散（去噪）
    
    Args:
        model: SDXL模型
        latents: 潜变量
        context: 文本嵌入
        t: 当前时间步
        guidance_scale: CFG比例
        added_cond_kwargs: 附加条件，必须包含'text_embeds'和'time_ids'
    """
    # Ensure latents are in correct dtype
    latents = latents.to(dtype=model.unet.dtype)
    latents_input = torch.cat([latents] * 2) if latents.shape[0] == context.shape[0] // 2 else latents
    
    # 确保added_cond_kwargs包含text_embeds
    if added_cond_kwargs is None or "text_embeds" not in added_cond_kwargs:
        raise ValueError("added_cond_kwargs must contain 'text_embeds' for SDXL")
        
    # 将所有输入转换为正确的dtype
    context = context.to(dtype=model.unet.dtype)
    processed_kwargs = {}
    for k, v in added_cond_kwargs.items():
        if isinstance(v, torch.Tensor):
            processed_kwargs[k] = v.to(dtype=model.unet.dtype)
        else:
            processed_kwargs[k] = v
    
    print("[DIFF] context shape:", context.shape)
    print("[DIFF] processed_kwargs['text_embeds'] shape:", processed_kwargs['text_embeds'].shape)
    print("[DIFF] latents_input shape:", latents_input.shape)
    
    # Add support for SDXL's additional conditioning
    noise_pred = model.unet(
        latents_input, 
        t, 
        encoder_hidden_states=context,
        added_cond_kwargs=processed_kwargs
    )["sample"]
    
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
    true_label = model.tokenizer.encode(imagenet_label.refined_Label[label.item()])
    target_label = model.tokenizer.encode(target_prompt) 
    print("decoder: ", true_label, target_label)

    """
            ==========================================
            ============ DDIM Inversion ==============
            === Details please refer to Appendix B ===
            ==========================================
    """
    latent, inversion_latents = ddim_reverse_sample(image, prompt, model,
                                                    num_inference_steps,
                                                    0, res=height)
    inversion_latents = inversion_latents[::-1]

    init_prompt = [prompt[0]]
    batch_size = len(init_prompt)
    latent = inversion_latents[start_step - 1]

    """
            ===============================================================================
            === Good initial reconstruction by optimizing the unconditional embeddings ====
            ======================= Details please refer to Section 3.4 ===================
            ===============================================================================
    """
    # 使用encode_prompt获取初始prompt的嵌入
    init_prompt_data = encode_prompt(model, init_prompt)
    
    # 分离uncond和cond嵌入以便优化
    null_prompt = [""] * batch_size
    null_data = encode_prompt(model, null_prompt)
    
    # 创建可训练的uncond embeddings
    uncond_embeddings = null_data["prompt_embeds"][:batch_size].clone().to(dtype=model_dtype)
    uncond_embeddings.requires_grad_(True)
    text_embeddings = init_prompt_data["prompt_embeds"][batch_size:].clone().to(dtype=model_dtype)
    
    # 用于SDXL条件的pooled embeddings
    uncond_pooled_embeds = null_data["pooled_prompt_embeds"][:batch_size].clone().to(dtype=model_dtype)
    text_pooled_embeds = init_prompt_data["pooled_prompt_embeds"][batch_size:].clone().to(dtype=model_dtype)

    all_uncond_emb = []
    all_uncond_pooled_emb = []  # 存储优化后的pooled embeddings
    latent = latent.to(dtype=model_dtype)
    latent, latents = init_latent(latent, model, height, width, batch_size)

    optimizer = optim.AdamW([uncond_embeddings], lr=1e-1)
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
            prompt_embeds = torch.cat([uncond_embeddings, text_embeddings])
            pooled_prompt_embeds = torch.cat([uncond_pooled_embeds, text_pooled_embeds])
            
            # SDXL的added_cond_kwargs
            added_cond_kwargs = {
                "text_embeds": pooled_prompt_embeds,  # [2*batch, 1280]
                "time_ids": time_ids[:2*batch_size]  # [2*batch, 6]
            }
            
            out_latents = diffusion_step(model, latents, prompt_embeds, t, guidance_scale, added_cond_kwargs)
            optimizer.zero_grad()
            target_latents = inversion_latents[start_step - 1 + ind + 1].to(dtype=model_dtype)
            loss = loss_func(out_latents, target_latents)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            # 使用更新后的uncond_embeddings
            prompt_embeds = torch.cat([uncond_embeddings, text_embeddings])
            pooled_prompt_embeds = torch.cat([uncond_pooled_embeds, text_pooled_embeds])
            
            added_cond_kwargs = {
                "text_embeds": pooled_prompt_embeds,  # [2*batch, 1280]
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
    prompt_data = encode_prompt(model, prompt)
    prompt_embeds = prompt_data["prompt_embeds"]
    pooled_prompt_embeds = prompt_data["pooled_prompt_embeds"]
    
    # 创建各个timestep的上下文
    contexts = []
    pooled_contexts = []
    for i in range(len(all_uncond_emb)):
        # 重复uncond embeddings以匹配batch size
        repeated_uncond = all_uncond_emb[i].repeat(batch_size, 1, 1)
        repeated_uncond_pooled = all_uncond_pooled_emb[i].repeat(batch_size, 1, 1)
        
        # 连接text embeddings
        current_context = torch.cat([repeated_uncond, prompt_embeds[batch_size:]])
        current_pooled = torch.cat([repeated_uncond_pooled, pooled_prompt_embeds[batch_size:]])
        
        contexts.append(current_context)
        pooled_contexts.append(current_pooled)

    # 准备SDXL条件参数
    time_ids_list = torch.tensor([*original_size, *crop_coords, *target_size], 
                              device=model.device, 
                              dtype=model_dtype).unsqueeze(0).repeat(batch_size * 2, 1)
    
    added_cond_kwargs_list = []
    for i in range(len(contexts)):
        added_cond_kwargs_list.append({
            "text_embeds": pooled_contexts[i],  # [2*batch, 1280]
            "time_ids": time_ids_list
        })

    original_latent = latent.clone()

    latent.requires_grad_(True)

    optimizer = optim.AdamW([latent], lr=1e-2)
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
        latents = torch.cat([original_latent, latent])

        for ind, t in enumerate(model.scheduler.timesteps[1 + start_step - 1:]):
            # 确保使用正确的上下文和条件参数
            current_context = contexts[ind]
            current_cond_kwargs = added_cond_kwargs_list[ind]
            latents = diffusion_step(model, latents, current_context, t, guidance_scale, current_cond_kwargs)

        before_attention_map = aggregate_attention(prompt, controller, args.res // 32, ("up", "down"), True, 0, is_cpu=False)
        after_attention_map = aggregate_attention(prompt, controller, args.res // 32, ("up", "down"), True, 1, is_cpu=False)

        before_true_label_attention_map = before_attention_map[:, :, 1: len(true_label) - 1]
        after_true_label_attention_map = after_attention_map[:, :, 1: len(true_label) - 1]

        if init_mask is None:
            # Convert attention map to float32 for image processing operations
            att_map = before_true_label_attention_map.detach().clone().float().mean(-1)
            att_map_max = att_map.max()
            # Avoid division by zero
            if att_map_max > 0:
                att_map = att_map / att_map_max
            else:
                att_map = torch.zeros_like(att_map)
                
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
        latents_for_decode = latents.to(dtype=model.vae.dtype)
        init_out_image = model.vae.decode(1 / 0.13025 * latents_for_decode)['sample'][1:] * init_mask + (1 - init_mask) * init_image

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

        # "Deceive" Strong Diffusion Model. Details please refer to Section 3.3
        variance_cross_attn_loss = after_true_label_attention_map.var() * args.cross_attn_loss_weight

        # Preserve Content Structure. Details please refer to Section 3.4
        self_attn_loss = controller.loss * args.self_attn_loss_weight

        loss = self_attn_loss + attack_loss + variance_cross_attn_loss

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
