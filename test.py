import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

model = build_sam3_image_model()
processor = Sam3Processor(model, amp_dtype=torch.bfloat16)
image = Image.open("shelf.jpg")
state = processor.set_image(image)
output = processor.set_text_prompt(state=state, prompt="individual product or empty space on shelf")
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
print(f"検出数: {len(boxes)}, スコア: {scores}")

# --- 可視化 ---
result = image.convert("RGBA")
overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
draw = ImageDraw.Draw(overlay)

masks_np = masks.squeeze(1).cpu().float().numpy()   # (N, H, W)
boxes_np = boxes.cpu().float().numpy()              # (N, 4) xyxy
scores_np = scores.cpu().float().numpy()

rng = np.random.default_rng(42)
colors = rng.integers(50, 230, size=(len(boxes_np), 3))

for i, (mask, box, score, color) in enumerate(zip(masks_np, boxes_np, scores_np, colors)):
    r, g, b = int(color[0]), int(color[1]), int(color[2])

    # マスクを半透明で塗る
    mask_img = Image.fromarray((mask * 120).astype(np.uint8))  # alpha=120
    colored = Image.new("RGBA", result.size, (r, g, b, 0))
    colored.putalpha(mask_img)
    overlay = Image.alpha_composite(overlay, colored)

    # バウンディングボックス
    x0, y0, x1, y1 = box
    draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0, 255), width=20)

    # スコアラベル
    label = f"{score:.2f}"
    draw.text((x0 + 2, y0 + 2), label, fill=(r, g, b, 255))

result = Image.alpha_composite(result, overlay).convert("RGB")
result.save("result.jpg")
print("result.jpg に保存しました")
