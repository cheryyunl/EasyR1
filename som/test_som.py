# 🎯 修复版SoM测试 - GUI-R1数据集已经是二进制格式
import pandas as pd
import io
from PIL import Image
from som_processor import SoMProcessor

# 读取数据
df = pd.read_parquet("/code/gui_r1_data/androidcontrol_high_test.parquet")

# 获取图像二进制数据（不需要base64解码！）
image_bytes = df.iloc[0]['image']['bytes']  # 直接使用bytes数据
original = Image.open(io.BytesIO(image_bytes)).convert('RGB')

print(f"📱 原图尺寸: {original.size}")

# SoM处理
processor = SoMProcessor()
som_image, coordinates = processor.process_image(original)

# 确保尺寸一致
som_image = som_image.resize(original.size) if som_image.size != original.size else som_image

# 保存结果
som_image.save("som_result.png")
original.save("original.png")

print(f"✅ 处理完成!")
print(f"📊 原图: {original.size}")
print(f"📊 SoM图: {som_image.size}")
print(f"🎯 标记数: {len(coordinates)}")

# 对比显示
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(original); ax1.set_title('Original'); ax1.axis('off')
ax2.imshow(som_image); ax2.set_title(f'SoM ({len(coordinates)} marks)'); ax2.axis('off')
plt.tight_layout()
plt.savefig("comparison.png", dpi=150, bbox_inches='tight')
print("💾 保存文件:")
print("   - original.png (原图)")
print("   - som_result.png (SoM处理结果)")  
print("   - comparison.png (对比图)")

# 显示一些标记信息
print(f"\n🎯 前5个标记的坐标:")
for i, (mark_id, coords) in enumerate(list(coordinates.items())[:5]):
    x, y, w, h = coords
    print(f"   Mark {mark_id}: ({x}, {y}) 尺寸: {w}×{h}")