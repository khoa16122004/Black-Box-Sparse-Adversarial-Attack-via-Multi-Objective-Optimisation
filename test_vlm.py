# Test saliency map for VLM/CLIP (MedCLIP) with prompt and visualize

import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import sys

try:
	import open_clip
except ImportError:
	open_clip = None

def get_openclip_model(model_name="ViT-B-32", pretrained="openai", device="cuda"):
	assert open_clip is not None, "open_clip not installed!"
	model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
	tokenizer = open_clip.get_tokenizer(model_name)
	model = model.to(device)
	model.eval()
	return model, preprocess, tokenizer

def compute_similarity_grad(
	model, image_tensor, prompt, method='simple', steps=20,
	encode_image=None, encode_text=None
):
	device = image_tensor.device
	image_tensor = image_tensor.clone().detach().unsqueeze(0)
	image_tensor.requires_grad_(True)
	# Encode text
	if encode_text is not None:
		text_emb = encode_text(prompt)
	else:
		text_emb = model.encode_text(texts=prompt)
	# Encode image
	if encode_image is not None:
		img_emb = encode_image(image_tensor)
	else:
		img_emb = model.encode_image(image_tensor)
	sim = torch.nn.functional.cosine_similarity(img_emb, text_emb)
	score = sim.sum()
	if method == 'simple':
		score.backward()
		grad = image_tensor.grad.detach()
		sal = grad.abs().sum(dim=1)
		sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
		return sal[0].cpu().numpy()
	elif method == 'ig':
		baseline = torch.zeros_like(image_tensor)
		grads = torch.zeros_like(image_tensor)
		for i in range(1, steps+1):
			alpha = float(i)/steps
			inp = baseline + alpha * (image_tensor - baseline)
			inp.requires_grad_(True)
			if encode_image is not None:
				img_emb = encode_image(inp)
			else:
				img_emb = model.encode_image(inp)
			sim = torch.nn.functional.cosine_similarity(img_emb, text_emb)
			score = sim.sum()
			score.backward()
			grads += inp.grad.detach()
		avg_grad = grads / steps
		ig = (image_tensor - baseline) * avg_grad
		sal = ig.abs().sum(dim=1)
		sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
		return sal[0].cpu().numpy()
	else:
		raise ValueError('Unknown method')

def test_vlm_saliency(image_path, prompt, method='simple',
					 model=None, preprocess=None, encode_image=None, encode_text=None):
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	# Load and preprocess image
	img = Image.open(image_path).convert('RGB')
	if preprocess is not None:
		img_tensor = preprocess(img).unsqueeze(0).to(device)
	else:
		img_tensor = torch.from_numpy(np.array(img).transpose(2,0,1)).float().div(255).unsqueeze(0).to(device)
	# Compute saliency
	sal = compute_similarity_grad(
		model, img_tensor[0], prompt, method=method,
		encode_image=encode_image, encode_text=encode_text
	)
	# Visualize
	img_np = np.array(img.resize((224,224))) / 255.0
	plt.figure(figsize=(10,4))
	plt.subplot(1,2,1)
	plt.imshow(img_np)
	plt.title('Input Image')
	plt.axis('off')
	plt.subplot(1,2,2)
	plt.imshow(img_np)
	plt.imshow(sal, cmap='jet', alpha=0.5)
	plt.title(f'Saliency ({method})')
	plt.axis('off')
	plt.suptitle(f'Prompt: {prompt}')
	plt.tight_layout()
	plt.show()

# Example usage for open_clip:
# if open_clip is not None:
#     model, preprocess, tokenizer = get_openclip_model()
#     def encode_image(x):
#         return model.encode_image(x)
#     def encode_text(t):
#         tokens = tokenizer([t]).to(next(model.parameters()).device)
#         return model.encode_text(tokens)
#     test_vlm_saliency('test_imgs/dog-and-cat-cover.jpg', 'a photo of dog', method='simple',
#                      model=model, preprocess=preprocess, encode_image=encode_image, encode_text=encode_text)
