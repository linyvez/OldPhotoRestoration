from turtle import color
import numpy as np
import cv2
from PIL import Image

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from preprocessing import preprocess_image
from contrast import contrast_process_image, apply_clahe
from colorization import colorize_image, auto_generate_grid_scribbles, BLENDING_FACTOR, RELAXATION_LIMIT
from damage import remove_scratches

def calculate_metrics(orig_rgb, result_rgb):
    p_val = psnr(orig_rgb, result_rgb)
    s_val = ssim(orig_rgb, result_rgb, channel_axis=2)
    
    orig_lab = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    res_lab = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    
    delta_e_map = np.sqrt(np.sum((orig_lab - res_lab)**2, axis=2))
    avg_delta_e = np.mean(delta_e_map)
    
    return p_val, s_val, avg_delta_e, delta_e_map


st.set_page_config(layout="wide")
st.title("Image Restoration")
st.sidebar.header("Colorization Parameters")

b_val = st.sidebar.slider("Blending Factor (b)", 1.0, 6.0, BLENDING_FACTOR)
rel_val = st.sidebar.slider("Relaxation Limit", 1.0, 10.0, RELAXATION_LIMIT)

tabs = st.tabs(["Full Photo Restoration", "Damage Correction", "Contrast Correction", "Manual Colorization", "Recovery", "Auto Colorization Evaluation"])

tab_full_correction = tabs[0]
tab_damage = tabs[1]
tab_contrast = tabs[2]
tab_manual_colorization = tabs[3]
tab_recovery_colorization = tabs[4]
tab_evaluation = tabs[5]

with tab_full_correction:
    st.header("Full Photo Restoration")
    st.write("Upload an **original** image. It will be grayscaled, then damage and contrast correction will be applied, and then manual colorization will start.")
    manual_file = st.file_uploader("Upload Original Image", type=["png", "jpg"], key="full_res_up")

    if st.button("New Image", key="full_new_img"):
        st.session_state.processed_gray = None
        st.experimental_rerun()

    if manual_file:
        raw_pil = Image.open(manual_file).convert("RGB")
        orig_color = np.array(raw_pil)
        gray_source = cv2.cvtColor(orig_color, cv2.COLOR_RGB2GRAY)

        preprocessed_img = preprocess_image(gray_source)

        if "processed_gray" not in st.session_state:
            st.session_state.processed_gray = None
        
        if st.button("Restore", key="full_restore"):
            restored_rgb, _ = remove_scratches(orig_color)
            damage_corrected = cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2GRAY)

            # Contrast Correction
            orig_img, orig_hist = contrast_process_image(damage_corrected)
            contrast_corrected, _ = apply_clahe(orig_img, orig_hist)

            st.session_state.processed_gray = contrast_corrected


        # Manual Colorization
        if st.session_state.processed_gray is not None:
            st.markdown("---")

            bg_image = Image.fromarray(st.session_state.processed_gray)
            h, w = st.session_state.processed_gray.shape

            stroke_color = st.color_picker("Pick Scribble Color", "#0000FF", key="full_st_color")
            stroke_width = st.slider("Brush Size", 1, 20, 5, key="full_st_width")
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0)",
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_image=bg_image,
                height=h, width=w,
                drawing_mode="freedraw",
                key="full_manual_canvas",
            )
            
            if st.button("Run Colorization", key="full_manual_btn"):
                if canvas_result.image_data is not None:
                    scribbles = {}
                    idx = np.where(canvas_result.image_data[:, :, 3] > 0)
                    for y, x in zip(idx[0], idx[1]):
                        color = tuple(canvas_result.image_data[y, x, :3])
                        if color not in scribbles: scribbles[color] = []
                        scribbles[color].append((y, x))
                    
                    if scribbles:
                        with st.spinner("Processing..."):
                            result = colorize_image(st.session_state.processed_gray, scribbles, blending_factor=b_val, relaxation=rel_val)
                            st.markdown("---")
                            st.subheader("Final Restored Photo")
                            st.image(result, caption="Colorized Result", width=600)
            

with tab_damage:
    st.header("Damage Correction")
    st.write("Upload an **original** image with scratches.")
    manual_file = st.file_uploader("Upload Original Image", type=["png", "jpg"], key="damage_up")
    
    if manual_file:
        orig_color = np.array(Image.open(manual_file).convert("RGB"))
        
        col_sets1, col_sets2 = st.columns(2)
        
        with col_sets1:
            thresh_val = st.slider("Threshold", 5, 100, 35, 
                                   help="Менше значення = більше знайдених подряпин.")
            
        with col_sets2:
            radius_val = st.slider("Inpaint Radius", 1, 10, 3,)

        c1, c2 = st.columns([1, 1])
        
        if st.button("Restore", key="damage_restore"):
            with c1:
                st.image(orig_color, caption="Original Image", use_column_width=True)
            
            with st.spinner("Processing scratches..."):
                restored_img, debug_mask = remove_scratches(orig_color, threshold=thresh_val, radius=radius_val)
            
            with c2:
                st.image(restored_img, caption="Restored Image", use_column_width=True)
            
            with st.expander("Show Scratch Mask"):
                st.image(debug_mask, caption="Mask", use_column_width=True)

with tab_contrast:
    st.header("Contrast Correction")
    st.write("Upload an **original** image. It will be grayscaled, then contrast correction will be applied.")
    manual_file = st.file_uploader("Upload Original Image", type=["png", "jpg"], key="contrast_up")
    if manual_file:
        orig_color = np.array(Image.open(manual_file).convert("RGB"))
        gray_source = cv2.cvtColor(orig_color, cv2.COLOR_RGB2GRAY)

        preprocessed_img = preprocess_image(gray_source)
        
        c1, c2 = st.columns([1, 1])
        if st.button("Restore", key="contrast_restore"):
            orig_img, orig_hist = contrast_process_image(preprocessed_img)
            with c1:
                st.image(orig_color, caption="Original Image", use_column_width=True)

            # correction
            contrast_corrected, _ = apply_clahe(orig_img, orig_hist)

            with c2:
                st.image(contrast_corrected, caption="Contrast Corrected Result", use_column_width=True)

with tab_manual_colorization:
    st.header("Manual Colorization with Scribbles")
    st.write("Upload a **gray-scale** image to start coloring.")
    manual_file = st.file_uploader("Upload Gray-scale Image", type=["png", "jpg"], key="manual_up")
    
    if manual_file:
        raw_pil = Image.open(manual_file).convert("RGB")
        gray_np = cv2.cvtColor(np.array(raw_pil), cv2.COLOR_RGB2GRAY)
        h, w = gray_np.shape
        
        c1, c2 = st.columns([1, 1])
        with c1:
            stroke_color = st.color_picker("Pick Scribble Color", "#0000FF", key="st_color")
            stroke_width = st.slider("Brush Size", 1, 20, 5, key="st_width")
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0)",
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_image=raw_pil,
                height=h, width=w,
                drawing_mode="freedraw",
                key="manual_canvas",
            )
        
        if st.button("Run Colorization", key="manual_btn"):
            if canvas_result.image_data is not None:
                scribbles = {}
                idx = np.where(canvas_result.image_data[:, :, 3] > 0)
                for y, x in zip(idx[0], idx[1]):
                    color = tuple(canvas_result.image_data[y, x, :3])
                    if color not in scribbles: scribbles[color] = []
                    scribbles[color].append((y, x))
                
                if scribbles:
                    with st.spinner("Processing..."):
                        result = colorize_image(gray_np, scribbles, blending_factor=b_val, relaxation=rel_val)
                        with c2:
                            st.image(result, caption="Colorized Result", use_column_width=True)

with tab_recovery_colorization:
    st.header("Selective Color Recovery")
    st.write("Scratch the grayscale image to reveal true color hints.")

    rev_file = st.file_uploader(
        "Upload Color Reference",
        type=["png", "jpg"],
        key="rev"
    )

    if rev_file:
        orig_np = np.array(Image.open(rev_file).convert("RGB"))
        gray_np = cv2.cvtColor(orig_np, cv2.COLOR_RGB2GRAY)
        h, w = gray_np.shape

        brush = st.slider(
            "Brush Size",
            4, 60, 12,
            key="rev_brush"
        )

        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Scratch Mask")

            canvas_width = 600
            scale = canvas_width / w
            disp_w = canvas_width
            disp_h = int(h * scale)

            gray_disp = Image.fromarray(gray_np).resize(
                (disp_w, disp_h),
                Image.BILINEAR
            )

            canvas = st_canvas(
                background_image=gray_disp,
                stroke_color="#FFFFFF",
                stroke_width=int(brush * scale),
                height=disp_h,
                width=disp_w,
                drawing_mode="freedraw",
                key="c_rev",
                display_toolbar=True,
            )

        if canvas.image_data is not None:
            mask_small = (canvas.image_data[:, :, 3] > 0).astype(np.uint8)

            mask_full = cv2.resize(
                mask_small,
                (w, h),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)

            reveal_preview = cv2.cvtColor(gray_np, cv2.COLOR_GRAY2RGB)
            reveal_preview[mask_full] = orig_np[mask_full]

            with c2:
                st.subheader("Recovered Color Hints (Live)")
                st.image(reveal_preview, use_column_width=True)

        if st.button("Propagate Colors", key="b_rev"):
            if canvas.image_data is None:
                st.warning("Draw on the image first.")
            else:
                mask_small = (canvas.image_data[:, :, 3] > 0).astype(np.uint8)
                mask_full = cv2.resize(
                    mask_small,
                    (w, h),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)

                def quantize(rgb, step=16):
                    return tuple((np.array(rgb) // step * step).astype(int))

                recovered_scribbles = {}
                ys, xs = np.where(mask_full)

                for y, x in zip(ys, xs):
                    q_rgb = quantize(orig_np[y, x])
                    if q_rgb not in recovered_scribbles:
                        recovered_scribbles[q_rgb] = []
                    recovered_scribbles[q_rgb].append((y, x))

                if not recovered_scribbles:
                    st.warning("No valid color hints found.")
                else:
                    with st.spinner("Propagating intrinsic color distances..."):
                        final_res = colorize_image(
                            gray_np,
                            recovered_scribbles,
                            blending_factor=b_val,
                            relaxation=rel_val
                        )

                    st.subheader("Final Colorized Result")
                    c_orig, c_res = st.columns(2)
                    with c_orig:
                        st.image(orig_np, caption="Original Image", use_column_width=True)
                    with c_res:
                        st.image(final_res, caption="Colorized Result", use_column_width=True)

with tab_evaluation:
    st.header("Auto-Colorization Metrics")
    st.write("Upload a **full-color** image. It will be grayscaled, then recolored using sampled grid points.")
    auto_file = st.file_uploader("Upload Full-Color Image", type=["png", "jpg"], key="auto_up")
    if auto_file:
        orig_color = np.array(Image.open(auto_file).convert("RGB"))
        gray_source = cv2.cvtColor(orig_color, cv2.COLOR_RGB2GRAY)
        
        num_auto = st.slider("Number of Sample Points", 1, 250, 20)
        
    if st.button("Calculate Metrics"):
        scribbles = auto_generate_grid_scribbles(orig_color, num_auto)
        
        scribble_viz = cv2.cvtColor(gray_source, cv2.COLOR_GRAY2RGB)
        for rgb_color, points in scribbles.items():
            color_tuple = tuple(int(v) for v in rgb_color)
            for (y, x) in points:
                cv2.circle(scribble_viz, (x, y), 2, color_tuple, -1)

        with st.spinner("Analyzing Color Accuracy..."):
            result = colorize_image(gray_source, scribbles, blending_factor=b_val)
            
            p_val, s_val, de_val, de_map = calculate_metrics(orig_color, result)
            
            m1, m2, m3 = st.columns(3)
            m1.metric("PSNR (Fidelity)", f"{p_val:.2f} dB")
            m2.metric("SSIM (Structure)", f"{s_val:.4f}")
            m3.metric("Avg ΔE (Perceptual Error)", f"{de_val:.2f}", delta_color="inverse")
            

            st.image([orig_color, scribble_viz, result], 
                    caption=["Original", "Scribble Map", "Colorized Result"], 
                    use_column_width=True)
            
            st.subheader("Perceptual Error Heatmap (Delta-E)")
            de_viz = np.clip(de_map, 0, np.percentile(de_map, 99))
            de_viz = ((de_viz.max() - de_viz) / de_viz.max() * 255).astype(np.uint8)
            de_heatmap = cv2.applyColorMap(de_viz, cv2.COLORMAP_JET)
            st.image(de_heatmap, caption="Hot areas = High Color Error", use_column_width=True)
