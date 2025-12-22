import numpy as np
import cv2
import heapq

BLENDING_FACTOR = 4.0
RELAXATION_LIMIT = 5.0

def colorize_image(gray_img, scribbles, blending_factor = BLENDING_FACTOR, relaxation=RELAXATION_LIMIT):
    H, W = gray_img.shape

    Y = gray_img.astype(np.float32)

    visit_counts = np.zeros((H, W), dtype=np.int32)
    unique_colors = list(scribbles.keys())
    dist_maps = np.full((len(unique_colors), H, W), np.inf, dtype=np.float32)

    pq = []
    for color_idx, color in enumerate(unique_colors):

        for (y, x) in scribbles[color]:
            dist_maps[color_idx, y, x] = 0
            heapq.heappush(pq, (0.0, y, x, color_idx))

    while pq:
        curr_dist, y, x, color_idx = heapq.heappop(pq)

        if visit_counts[y, x] >= relaxation:
                    continue

        if curr_dist > dist_maps[color_idx, y, x]:
            continue

        visit_counts[y, x] += 1

        for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy

            if not (0 <= nx < W and 0 <= ny < H):
                continue

            weight = abs(Y[ny, nx] - Y[y, x])
            new_dist = curr_dist + weight

            if new_dist < dist_maps[color_idx, ny, nx]:
                dist_maps[color_idx, ny, nx] = new_dist
                heapq.heappush(pq, (new_dist, ny, nx, color_idx))
    
    weights = 1/((dist_maps + 1e-5) ** blending_factor)
    weights_sum = np.sum(weights, axis=0)

    final_chroma = np.zeros((H, W, 2), dtype=np.float32)

    for color_idx, color in enumerate(unique_colors):
        pixel = np.array([[color]], dtype=np.uint8)
        ycc = cv2.cvtColor(pixel, cv2.COLOR_RGB2YCrCb)[0, 0]

        Cr = float(ycc[1])
        Cb = float(ycc[2])

        w = weights[color_idx] / weights_sum

        final_chroma[:, :, 0] += w * Cr
        final_chroma[:, :, 1] += w * Cb


    for color in scribbles:
        pixel = np.array([[color]], dtype=np.uint8)
        ycc = cv2.cvtColor(pixel, cv2.COLOR_RGB2YCrCb)[0, 0]

        Cr, Cb = float(ycc[1]), float(ycc[2])

        for (y, x) in scribbles[color]:
            final_chroma[y, x, 0] = Cr
            final_chroma[y, x, 1] = Cb

    result_ycrcb = np.zeros((H, W, 3), dtype=np.float32)
    result_ycrcb[:, :, 0] = Y
    result_ycrcb[:, :, 1] = final_chroma[:, :, 0]
    result_ycrcb[:, :, 2] = final_chroma[:, :, 1]

    result_final = np.clip(result_ycrcb, 0, 255).astype(np.uint8)

    return cv2.cvtColor(result_final, cv2.COLOR_YCrCb2RGB)

def auto_generate_grid_scribbles(color_img, num_scribbles, patch_size=9, quant_step=16):
    H, W, _ = color_img.shape
    half_p = patch_size // 2
    
    gray = cv2.cvtColor(color_img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    variation_map = np.sqrt(grad_x**2 + grad_y**2)
    variation_map += 1e-5

    variation_flat = variation_map.flatten()
    variation_flat /= variation_flat.sum()
    
    scribbles = {}

    def quantize_color(rgb):
        return tuple((np.array(rgb) // quant_step * quant_step).astype(int))
    
    count_patches = 0
    max_attempts = num_scribbles * 5
    
    while count_patches < num_scribbles and max_attempts > 0:
        max_attempts -= 1
        idx = np.random.choice(H * W, p=variation_flat)
        y_center, x_center = divmod(idx, W)
        
        if not (half_p <= y_center < H - half_p and half_p <= x_center < W - half_p):
            continue

        for i in range(-half_p, half_p + 1):
            for j in range(-half_p, half_p + 1):
                curr_y, curr_x = y_center + i, x_center + j
                q_color = quantize_color(color_img[curr_y, curr_x])
                if q_color not in scribbles:
                    scribbles[q_color] = []
                scribbles[q_color].append((curr_y, curr_x))
        
        count_patches += 1

    return scribbles
