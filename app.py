import sys
import subprocess
import os
# 强制使用 headless 版本的 OpenCV，避免服务器环境 libGL 依赖问题
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
try:
    import cv2
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless", "--force-reinstall", "-q"])
    import cv2

import streamlit as st
import numpy as np
import pandas as pd
from ultralytics import YOLO
from PIL import Image
import io

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="精米尺寸测量与品种识别系统",
    page_icon="🌾",
    layout="wide"
)

# ==================== 标题区 ====================
st.title("🌾 精米尺寸测量与品种识别系统")
st.markdown("基于 YOLOv8 实例分割的米粒长宽测量、周长计算与品种匹配")
st.divider()

# ==================== 侧边栏配置 ====================
with st.sidebar:
    st.header("⚙️ 系统设置")
    
    # 模型路径配置
    st.subheader("模型配置")
    model_path = st.text_input(
        "模型权重路径",
        value="best.pt",
        help="训练好的 YOLOv8-seg 模型权重文件路径"
    )
    
    # 像素换算配置
    st.subheader("尺寸校准")
    pixel_per_mm = st.number_input(
        "每毫米对应像素数",
        min_value=1.0,
        max_value=1000.0,
        value=50.0,
        step=1.0,
        help="根据拍摄设备和分辨率校准，可用已知尺寸的参照物标定"
    )
    
    # 置信度阈值
    conf_threshold = st.slider(
        "检测置信度阈值",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05
    )
    
    st.divider()
    st.caption("💡 提示：首次使用请先校准像素-毫米换算比例")

# ==================== 加载模型（缓存） ====================
@st.cache_resource
def load_model(model_path):
    if os.path.exists(model_path):
        return YOLO(model_path)
    return None

model = load_model(model_path)

# ==================== 品种基准库加载 ====================
@st.cache_data
def load_variety_db(csv_path):
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

# 尝试加载品种库
variety_db = load_variety_db("variety_db.csv")

# ==================== 核心函数：米粒尺寸计算 ====================
def calculate_rice_metrics(mask, pixel_per_mm):
    """
    从分割掩码计算单颗米粒的尺寸指标
    返回：粒长(mm)、粒宽(mm)、长宽比、周长(mm)、面积(mm²)
    """
    # 转换为uint8格式
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # 查找轮廓
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    contour = max(contours, key=cv2.contourArea)
    
    # 计算最小外接矩形（用于粒长粒宽）
    rect = cv2.minAreaRect(contour)
    (x, y), (w, h), angle = rect
    length_px = max(w, h)
    width_px = min(w, h)
    
    # 转换为毫米
    length_mm = length_px / pixel_per_mm
    width_mm = width_px / pixel_per_mm
    
    # 长宽比
    aspect_ratio = length_mm / width_mm if width_mm > 0 else 0
    
    # 周长（毫米）
    perimeter_px = cv2.arcLength(contour, True)
    perimeter_mm = perimeter_px / pixel_per_mm
    
    # 面积（平方毫米）
    area_px = cv2.contourArea(contour)
    area_mm2 = area_px / (pixel_per_mm ** 2)
    
    return {
        "粒长(mm)": round(length_mm, 2),
        "粒宽(mm)": round(width_mm, 2),
        "长宽比": round(aspect_ratio, 2),
        "周长(mm)": round(perimeter_mm, 2),
        "面积(mm²)": round(area_mm2, 2)
    }

# ==================== 品种匹配函数 ====================
def match_variety(avg_length, avg_width, avg_ratio, variety_db):
    """
    基于平均尺寸匹配最接近的品种
    """
    if variety_db is None or variety_db.empty:
        return None
    
    # 兼容新旧列名
    length_col = '粒长(mm)' if '粒长(mm)' in variety_db.columns else '粒长'
    width_col = '粒宽(mm)' if '粒宽(mm)' in variety_db.columns else '粒宽'
    code_col = '品种编号' if '品种编号' in variety_db.columns else None
    source_col = '数据来源' if '数据来源' in variety_db.columns else None
    
    # 计算与每个品种的欧氏距离（归一化）
    scores = []
    for _, row in variety_db.iterrows():
        dist = np.sqrt(
            ((avg_length - row.get(length_col, 0)) / 10) ** 2 +
            ((avg_width - row.get(width_col, 0)) / 5) ** 2 +
            ((avg_ratio - row.get('长宽比', 0)) / 2) ** 2
        )
        name = row.get('品种名称', '未知')
        code = row.get(code_col, '') if code_col else ''
        source = row.get(source_col, '') if source_col else ''
        scores.append((name, code, source, dist))
    
    # 按距离排序，取前5
    scores.sort(key=lambda x: x[3])
    return scores[:5]

# ==================== 主界面：图片上传 ====================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📷 上传米粒图片")
    uploaded_file = st.file_uploader(
        "选择图片文件",
        type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"],
        help="支持常见图片格式，建议使用平整背景、均匀光照的图片"
    )
    
    if uploaded_file is not None:
        # 读取图片
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        st.image(image_rgb, caption="原始图片", use_container_width=True)

# ==================== 分析按钮与结果展示 ====================
with col2:
    st.subheader("🔍 分析结果")
    
    if uploaded_file is not None:
        if model is None:
            st.error(f"❌ 未找到模型文件：{model_path}")
            st.info("请将训练好的 best.pt 模型文件放在应用同目录下，或在侧边栏修改路径")
        else:
            if st.button("🚀 开始分析", type="primary", use_container_width=True):
                with st.spinner("正在进行米粒分割与尺寸计算..."):
                    # 模型推理
                    results = model.predict(
                        image,
                        conf=conf_threshold,
                        verbose=False
                    )
                    
                    result = results[0]
                    
                    # 检查是否有检测结果
                    if result.masks is None or len(result.masks) == 0:
                        st.warning("⚠️ 未检测到米粒，请调整图片或降低置信度阈值")
                    else:
                        # 绘制分割结果
                        result_image = result.plot()
                        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                        
                        st.image(result_image_rgb, caption=f"分割结果（共检测到 {len(result.masks)} 颗米粒）", use_container_width=True)
                        
                        # 计算每颗米粒的尺寸
                        all_metrics = []
                        for i, mask in enumerate(result.masks.data.cpu().numpy()):
                            # 将mask resize到原图尺寸
                            mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
                            metrics = calculate_rice_metrics(mask_resized, pixel_per_mm)
                            if metrics:
                                metrics["编号"] = i + 1
                                all_metrics.append(metrics)
                        
                        if all_metrics:
                            df_metrics = pd.DataFrame(all_metrics)
                            df_metrics = df_metrics[["编号", "粒长(mm)", "粒宽(mm)", "长宽比", "周长(mm)", "面积(mm²)"]]
                            
                            # 统计信息
                            st.subheader("📊 尺寸统计")
                            col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
                            with col_stats1:
                                st.metric("米粒数量", len(all_metrics))
                            with col_stats2:
                                st.metric("平均粒长", f"{df_metrics['粒长(mm)'].mean():.2f} mm")
                            with col_stats3:
                                st.metric("平均粒宽", f"{df_metrics['粒宽(mm)'].mean():.2f} mm")
                            with col_stats4:
                                st.metric("平均长宽比", f"{df_metrics['长宽比'].mean():.2f}")
                            
                            # 详细数据表
                            with st.expander("📋 查看每颗米粒详细数据"):
                                st.dataframe(df_metrics, use_container_width=True)
                            
                            # 品种匹配
                            if variety_db is not None and not variety_db.empty:
                                st.subheader("🏷️ 品种匹配结果")
                                avg_length = df_metrics['粒长(mm)'].mean()
                                avg_width = df_metrics['粒宽(mm)'].mean()
                                avg_ratio = df_metrics['长宽比'].mean()
                                
                                matches = match_variety(avg_length, avg_width, avg_ratio, variety_db)
                                if matches:
                                    for rank, (name, code, source, dist) in enumerate(matches, 1):
                                        similarity = max(0, 100 - dist * 20)
                                        code_str = f"（{code}）" if code else ""
                                        source_str = f" · {source}" if source else ""
                                        st.markdown(f"**第{rank}名**：{name}{code_str}（匹配度：{similarity:.1f}%{source_str}）")
                                
                                # 显示当前测量值供参考
                                st.caption(f"📏 当前测量：粒长 {avg_length:.2f}mm / 粒宽 {avg_width:.2f}mm / 长宽比 {avg_ratio:.2f}")
                                st.caption(f"📚 品种库共 {len(variety_db)} 个品种，尺寸数据为参考值，建议用实测数据替换以提高匹配精度")
                            else:
                                st.info("ℹ️ 未检测到品种数据库文件（variety_db.csv），如需品种匹配功能请添加该文件")
                            
                            # 导出结果
                            st.subheader("💾 导出结果")
                            csv = df_metrics.to_csv(index=False).encode('utf-8-sig')
                            st.download_button(
                                "📥 下载尺寸数据 CSV",
                                data=csv,
                                file_name="米粒尺寸测量结果.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                            
                            # 保存结果图
                            _, buffer = cv2.imencode('.png', result_image)
                            st.download_button(
                                "📥 下载分割结果图",
                                data=buffer.tobytes(),
                                file_name="米粒分割结果.png",
                                mime="image/png",
                                use_container_width=True
                            )
    else:
        st.info("👆 请先在左侧上传米粒图片")

# ==================== 底部说明 ====================
st.divider()
st.markdown("""
### 📖 使用说明
1. **模型准备**：将训练好的 `best.pt` 模型文件放在应用同目录
2. **尺寸校准**：在侧边栏设置正确的像素-毫米换算比例（可用已知尺寸参照物标定）
3. **上传图片**：选择平整背景、米粒分散不重叠的图片
4. **开始分析**：点击按钮自动完成分割、测量和品种匹配
5. **导出结果**：支持下载尺寸数据表和分割结果图

### ⚠️ 注意事项
- 米粒尽量分散摆放，避免重叠遮挡
- 保持光照均匀，背景干净
- 首次使用请务必校准像素换算比例，否则尺寸数据不准确
""")

