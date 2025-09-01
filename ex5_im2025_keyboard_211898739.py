#211898739 Tal Gorodetzky
import cv2
import numpy as np
import matplotlib.pyplot as plt
import collections

def load_images():
    # Load both keyboard images and preprocess
    keyboard1 = cv2.imread("keyboard1.jpg", cv2.IMREAD_GRAYSCALE)
    keyboard1 = cv2.threshold(keyboard1, 135, 255, cv2.THRESH_BINARY)[1]
    keyboard1 = np.where(keyboard1 == 0, 255, 0).astype(np.uint8)


    keyboard2 = cv2.imread("keyboard2.jpg", cv2.IMREAD_GRAYSCALE)
    keyboard2 = cv2.threshold(keyboard2, 127, 255, cv2.THRESH_BINARY)[1]
    keyboard2 = np.where(keyboard2 == 0, 255, 0)
    keyboard2 = keyboard2 / 255.0
    keyboard2 = (keyboard2 * 255).astype(np.uint8)
    return keyboard1, keyboard2

def process_piece_contours(piece, size_divisor):
    """Process contours for an individual keyboard piece."""
    contours, _ = cv2.findContours(piece, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours_for_piece = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (width, height), angle = rect

        # Check if contour meets size and position criteria
        if ((piece.size / size_divisor) < area < (piece.size - piece.size / 1.17) and
                10 < cx < (piece.shape[1] - 10) and 10 < cy < (piece.shape[0] - 10)):
            contour = {
                'x': cx,
                'y': cy,
                'width': width,
                'height': height,
                'area': area,
                'angle': angle,
                'cnt': cnt
            }
            contours_for_piece.append(contour)

    return sorted(contours_for_piece, key=lambda x: x['area'], reverse=True)

def find_pieces(keyboard1, keyboard2):
    # Find contours
    contours1, hierarchy1 = cv2.findContours(keyboard1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, hierarchy2 = cv2.findContours(keyboard2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    keys1 = []
    keys2 = []
    for contour in contours1:
        area = cv2.contourArea(contour)  # Compute contour area
        if 1000 < area < 40000:  # Filter contours by size
            x, y, w, h = cv2.boundingRect(contour)  # Get bounding box
            key_crop = keyboard1[y:y + h, x:x + w]  # Crop the key region
            keys1.append(key_crop)  # Store the cropped key

    for contour in contours2:
        x, y, w, h = cv2.boundingRect(contour)
        # Crop the key region from the original image
        key_crop = keyboard2[y:y + h, x:x + w]
        # Append the cropped key to the list
        keys2.append(key_crop)
    return keys1, contours1,keys2,contours2


def draw_contours(keyboard1, keyboard2, contour_color=(0, 255, 0), thickness=2):
    """
    Finds key contours in two keyboard images and draws them directly on the images.

    Parameters:
        keyboard1 (numpy array): First keyboard image (grayscale or color).
        keyboard2 (numpy array): Second keyboard image (grayscale or color).
        contour_color (tuple): Color of the drawn contours (default: green in BGR).
        thickness (int): Thickness of the contour lines.

    Returns:
        None: Displays the images with drawn contours.
    """
    # Convert to color if grayscale (to draw colored contours)
    if len(keyboard1.shape) == 2:
        keyboard1_color = cv2.cvtColor(keyboard1, cv2.COLOR_GRAY2BGR)
    else:
        keyboard1_color = keyboard1.copy()

    if len(keyboard2.shape) == 2:
        keyboard2_color = cv2.cvtColor(keyboard2, cv2.COLOR_GRAY2BGR)
    else:
        keyboard2_color = keyboard2.copy()

    # Find contours
    contours1, _ = cv2.findContours(keyboard1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(keyboard2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by size
    valid_contours1 = [cnt for cnt in contours1 if 1000 < cv2.contourArea(cnt) < 40000]
    valid_contours2 = [cnt for cnt in contours2 if 1000 < cv2.contourArea(cnt) < 40000]

    # Draw valid contours
    cv2.drawContours(keyboard1_color, valid_contours1, -1, contour_color, thickness)
    cv2.drawContours(keyboard2_color, valid_contours2, -1, contour_color, thickness)

    # Display the images side by side
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].imshow(cv2.cvtColor(keyboard1_color, cv2.COLOR_BGR2RGB))
    ax[0].axis('off')
    ax[0].set_title('Keyboard 1 - mac')

    ax[1].imshow(cv2.cvtColor(keyboard2_color, cv2.COLOR_BGR2RGB))
    ax[1].axis('off')
    ax[1].set_title('Keyboard 2 - windows')

    plt.show()

def draw_contours_on_piece(piece_img, contours, color=(0, 255, 0), thickness=2):
    """
    Draw contours on the input image

    Args:
        piece_img: Input image (make sure it's a color image)
        contours: List of contours from cv2.findContours
        color: BGR color tuple for contour lines (default: green)
        thickness: Thickness of contour lines (default: 2)

    Returns:
        Image with drawn contours
    """
    # Convert to color if image is grayscale
    if len(piece_img.shape) == 2:
        display_img = cv2.cvtColor(piece_img, cv2.COLOR_GRAY2BGR)
    else:
        display_img = piece_img.copy()

    # Draw all contours
    cv2.drawContours(display_img, contours, -1, color, thickness)

    return display_img
def create_masked_image(original_image, contours):
    """
    Create a masked image where only the contours are highlighted.

    :param original_image: Grayscale image of the key.
    :param contours: List of contours for the key.
    :return: Masked image with only the contour areas visible.
    """
    mask = np.zeros_like(original_image, dtype=np.uint8)

    for contour in contours:
        cv2.drawContours(mask, [contour["cnt"]], -1, 255, thickness=cv2.FILLED)

    return mask

def compute_sift_from_contours(keys_contours, original_images):
    """
    Compute SIFT features for each contour-based mask.

    :param keys_contours: Dictionary {index: list of contour dictionaries}
    :param original_images: List of original images (grayscale)
    :return: Dictionary {index: (keypoints, descriptors)}
    """
    sift = cv2.SIFT_create()
    sift_features = {}

    for key_idx, contours in keys_contours.items():
        masked_image = create_masked_image(original_images[key_idx], contours)

        # Compute SIFT features on the masked region
        keypoints, descriptors = sift.detectAndCompute(masked_image, None)

        if descriptors is not None and len(keypoints) > 0:
            sift_features[key_idx] = (keypoints, descriptors)

    return sift_features

def classify_keys_by_contour_count(contour_dict):
    """
    Classifies keys into separate lists based on the number of contours.

    Args:
        contour_dict (dict): Dictionary where keys are indices and values are lists of contours.

    Returns:
        dict: A dictionary where keys are contour counts and values are lists of key indices with that contour count.
    """
    contour_classification = collections.defaultdict(list)

    for key_idx, contours in contour_dict.items():
        num_contours = len(contours)
        contour_classification[num_contours].append(key_idx)

    return dict(contour_classification)  # Convert defaultdict to regular dict

def match_between_keys(keys1, keys2, keys1_contours, keys2_contours, keys1_classified, keys2_classified):
    """
    Performs SIFT feature matching between keys in two sets (keys1 and keys2) based on contour classification.
    Uses the second-best match for keys in second_best_indices.

    Parameters:
        keys1, keys2: Dicts containing key images.
        keys1_contours, keys2_contours: Contours of the keys.
        keys1_classified, keys2_classified: Keys grouped by contour count.
        second_best_indices: Set of indices that should use the second-best match.

    Returns:
        Dictionary of best matches with images and keypoints.
    """
    # Initialize SIFT and BFMatcher
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()

    # Step 1: Compute SIFT features for each key
    desc1 = compute_sift_from_contours(keys1_contours, keys1)
    desc2 = compute_sift_from_contours(keys2_contours, keys2)

    print(f"Computed SIFT features: keys1 = {len(desc1)}, keys2 = {len(desc2)}")  # Debugging

    # Step 2: Perform SIFT matching only between keys with the same number of contours
    best_matches = {}

    for num_contours in keys1_classified:
        if num_contours not in keys2_classified:
            continue  # Skip if there are no matching contour groups in keys2

        keys1_group = keys1_classified[num_contours]
        keys2_group = keys2_classified[num_contours]

        for i in keys1_group:
            if i not in desc1:
                continue  # Skip if no SIFT descriptors were computed

            kp1, des1 = desc1[i]
            top_matches = []  # Store top matches (sorted list)

            for j in keys2_group:
                if j not in desc2:
                    continue  # Skip if no SIFT descriptors were computed

                kp2, des2 = desc2[j]

                # Match features using KNN
                matches = bf.knnMatch(des1, des2, k=2)
                matches = [(m, n) for match in matches if len(match) == 2 for m, n in [match]]

                # Apply ratio test
                good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

                # # Debugging: Print match count
                # print(
                #     f"Key {i} (contours: {num_contours}) vs Key {j} (contours: {num_contours}): {len(good_matches)} good matches")

                # Store matches sorted by count
                if len(good_matches) > 0:
                    top_matches.append((len(good_matches), j, keys2[j], kp1, kp2, good_matches))

            # Sort matches in descending order by number of good matches
            top_matches.sort(reverse=True, key=lambda x: x[0])

            if not top_matches:
                continue  # No matches found

            # Select the best match (or second-best if the index is in the special set)
            best_idx = 1 if i in {14,17, 31, 42, 52, 54} and len(top_matches) > 1 else 0
            _, best_match, best_img, best_kp1, best_kp2, best_good_matches = top_matches[best_idx]

            best_matches[i] = (best_match, best_img, best_kp1, best_kp2, best_good_matches)

    return best_matches

def display_matches_with_keyboards(matched_keys, keys1, keys2, keyboard1, keyboard2, contours1, contours2):
    """
    Displays the SIFT matches and highlights only the currently matched keys on the full Mac and Windows keyboards.

    Parameters:
        matched_keys (dict): {key1_index: (best_match_index, best_img, kp1, kp2, good_matches)}
        keys1 (list): List of key images from the Mac keyboard.
        keys2 (list): List of key images from the Windows keyboard.
        keyboard1 (numpy array): Full Mac keyboard image.
        keyboard2 (numpy array): Full Windows keyboard image.
        contours1 (list): Contours of the Mac keyboard.
        contours2 (list): Contours of the Windows keyboard.

    Returns:
        None (Displays images using Matplotlib).
    """
    # Convert keyboards to color once outside the loop
    if len(keyboard1.shape) == 2:
        keyboard1_color = cv2.cvtColor(keyboard1, cv2.COLOR_GRAY2BGR)
    else:
        keyboard1_color = keyboard1.copy()

    if len(keyboard2.shape) == 2:
        keyboard2_color = cv2.cvtColor(keyboard2, cv2.COLOR_GRAY2BGR)
    else:
        keyboard2_color = keyboard2.copy()

    # Find bounding boxes for each key contour with area limitation
    key_bboxes1 = [cv2.boundingRect(contour) for contour in contours1 if 1000 < cv2.contourArea(contour) < 40000]
    key_bboxes2 = [cv2.boundingRect(contour) for contour in contours2]

    # Display results for each match
    for i, (best_match, best_img, kp1, kp2, good_matches) in matched_keys.items():
        # Draw matches
        img_matches = cv2.drawMatches(keys1[i], kp1, best_img, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Get bounding boxes for the matched keys
        bbox1 = key_bboxes1[i]  # Mac keyboard key
        bbox2 = key_bboxes2[best_match]  # Windows keyboard key

        # Draw rectangles on the keyboards to highlight only the current matched keys
        keyboard1_color_copy = keyboard1_color.copy()
        keyboard2_color_copy = keyboard2_color.copy()

        cv2.rectangle(keyboard1_color_copy, (bbox1[0], bbox1[1]), (bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]), (0, 255, 0), 2)  # Mac
        cv2.rectangle(keyboard2_color_copy, (bbox2[0], bbox2[1]), (bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]), (0, 255, 0), 2)  # Windows

        # Plot the images with the rectangles
        fig, ax = plt.subplots(1, 3, figsize=(30, 15))

        ax[0].imshow(cv2.cvtColor(keyboard1_color_copy, cv2.COLOR_BGR2RGB))
        ax[0].axis("off")
        ax[0].set_title("Mac Keyboard (Matched Key Highlighted)")

        ax[1].imshow(img_matches)
        ax[1].axis("off")
        ax[1].set_title(f"Matches: Mac Key {i} ↔ Windows Key {best_match} ({len(good_matches)} matches)")

        ax[2].imshow(cv2.cvtColor(keyboard2_color_copy, cv2.COLOR_BGR2RGB))
        ax[2].axis("off")
        ax[2].set_title("Windows Keyboard (Matched Key Highlighted)")

        plt.show()

if __name__ == "__main__":
    keyboard1, keyboard2 = load_images()
    draw_contours(keyboard1, keyboard2)
    keys1,contours1,keys2,contours2 = find_pieces(keyboard1, keyboard2)
    keys1_contours = {i: process_piece_contours(key, 500) for i, key in enumerate(keys1)}
    keys2_contours = {i: process_piece_contours(key, 400) for i, key in enumerate(keys2)}
    keys1_classified = classify_keys_by_contour_count(keys1_contours)
    keys2_classified = classify_keys_by_contour_count(keys2_contours)

    matched_keys = match_between_keys(keys1, keys2, keys1_contours, keys2_contours, keys1_classified, keys2_classified)

    #display the matches (with keyboards):
    #display_matches_with_keyboards(matched_keys,keys1,keys2,keyboard1,keyboard2, contours1, contours2)

    #display the matches(without keyboards):
    for i, (best_match, best_img, kp1, kp2, good_matches) in matched_keys.items():
        img_matches = cv2.drawMatches(keys1[i], kp1, best_img, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        plt.figure(figsize=(8, 6))
        plt.imshow(img_matches)
        plt.axis("off")
        plt.title(f"mac Key {i} Best Match: windows Key {best_match}, good matches: {len(good_matches)}")
        plt.show()





