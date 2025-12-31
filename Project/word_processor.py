import numpy as np
from PIL import Image, ImageFilter

def process_word_image(image_file):
    """
    Segment a word image into characters and resize each to 16x16.
    Returns: List of dictionaries {'bbox': [x,y,w,h], 'pixels': list}
    """
    # Load and convert to grayscale
    img = Image.open(image_file).convert('L')
    img_arr = np.array(img)
    
    # Threshold (assuming drawing is light on dark or dark on light)
    # The canvas sends black background with gold/white drawing?
    # Actually format is usually RGBA.
    # index.html sends base64?
    # We will assume app.py handles base64 to BytesIO.
    
    # Check contrast. If mostly dark, assume background is 0.
    if np.mean(img_arr) < 128:
        # Dark background, bright text
        binary = (img_arr > 30).astype(np.uint8)
    else:
        # Light background, dark text
        binary = (img_arr < 200).astype(np.uint8)

    # 1. Projection Profile (Vertical Sums)
    h, w = binary.shape
    col_sums = np.sum(binary, axis=0)
    
    # Dynamic Threshold
    threshold = 1 # At least 1 pixel
    
    segments = []
    in_char = False
    start_x = 0
    
    # Dilation to bridge gaps (simulated)
    # Simple 1D dilation of col_sums
    dilated_sums = np.copy(col_sums)
    for i in range(1, w-1):
        dilated_sums[i] = max(col_sums[i-1], col_sums[i], col_sums[i+1])
        
    for x in range(w):
        if not in_char and dilated_sums[x] > threshold:
            in_char = True
            start_x = x
        elif in_char and dilated_sums[x] <= threshold:
            # Check for real gap (lookahead)
            is_gap = True
            # Reduced lookahead from 5 to 1 to prevent bridging separate letters
            if x + 1 < w:
                if np.any(dilated_sums[x:x+1] > threshold):
                    is_gap = False
            
            if is_gap:
                in_char = False
                segments.append((start_x, x))
                
    if in_char:
        segments.append((start_x, w))
        
    # Merge small gaps (Logic from Frontend)
    merged = []
    if segments:
        curr_start, curr_end = segments[0]
        for i in range(1, len(segments)):
            next_start, next_end = segments[i]
            # Reduced merge threshold from 3 to 1 to keep letters separate
            if next_start - curr_end < 1: 
                curr_end = next_end
            else:
                merged.append((curr_start, curr_end))
                curr_start, curr_end = next_start, next_end
        merged.append((curr_start, curr_end))
        
    results = []
    for start_x, end_x in merged:
        if end_x - start_x < 5: continue # Skip noise
        
        # Extract char column
        char_col = binary[:, start_x:end_x]
        
        # Find Y bounds
        row_sums = np.sum(char_col, axis=1)
        y_indices = np.where(row_sums > 0)[0]
        if len(y_indices) == 0: continue
        min_y, max_y = y_indices[0], y_indices[-1]
        
        # Crop tight
        char_crop = char_col[min_y:max_y+1, :]
        ch_h, ch_w = char_crop.shape
        
        # Make Square Canvas (preserve aspect ratio)
        dim = max(ch_h, ch_w)
        square = np.zeros((dim, dim), dtype=np.uint8)
        
        # Center
        off_y = (dim - ch_h) // 2
        off_x = (dim - ch_w) // 2
        square[off_y:off_y+ch_h, off_x:off_x+ch_w] = char_crop
        
        # Create PIL image from square
        pil_sq = Image.fromarray(square * 255)
        
        # Apply Bold Filter (MaxFilter) on HIGH RES image to preserve shape
        # This thickens the strokes naturally before downscaling
        # Apply Bold Filter (MaxFilter) - DISABLED to match Training (Draw) data
        # if dim > 20: 
        #      pil_sq = pil_sq.filter(ImageFilter.MaxFilter(3))
        
        # Resize to 16x16 using PIL (Lanczos/Bicubic for smoothness)
        pil_tiny = pil_sq.resize((16, 16), Image.Resampling.LANCZOS)
        
        # Normalize back to 0-1 for SNN
        tiny_arr = np.array(pil_tiny) / 255.0
        
        # Flatten
        pixels = tiny_arr.flatten().tolist()
        
        results.append({
            'bbox': [int(start_x), int(min_y), int(end_x-start_x), int(max_y-min_y+1)],
            'pixels': pixels
        })
        
    return results
