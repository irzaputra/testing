import os
import shutil
import cv2
import imutils
import numpy as np
import concurrent
import pytesseract
from PIL import Image
from skimage.filters import threshold_local
from ultralytics import YOLO

#mengembalikan jumlah hasil + hasil sebagai kotak
def detect_with_yolo(preloaded_model: YOLO, image: Image, verbose: bool)-> (int, any): # type: ignore
    result = preloaded_model.predict(image,verbose=verbose)[0]
    return (len(result.boxes), result.boxes)

def normalize_label(label):
    return label.strip().lower()

def clean_plate_into_contrours(plate_img:np.ndarray, fixed_width: int) -> np.ndarray:
    plate_img = cv2.GaussianBlur(plate_img,(5,5),0)
    V = cv2.split(cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV))[2]
    T = threshold_local(V, 99 ,offset=5, method='gaussian')
    thresh = (V > T).astype ('uint8') * 255
    thresh = cv2.bitwise_not(thresh)
    plate_img = imutils.resize(plate_img, width=fixed_width)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)
    return thresh

def get_letter_rectangles_from_contours(iwl):
    contours,_ = cv2.findContours(iwl,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    rectangles = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if(h < (iwl.shape[0] / 5) or w > (iwl.shape[1] / 5)):
            continue
        rectangles.append([x,y,w,h])
    final_rect= []
    for (x,y,w,h) in rectangles:
        flag = True
        for (x2,y2,w2,h2) in rectangles:
            if x > x2 and y > y2 and x + w < x2 + w2 and y + h < y2 + h2:
                flag = False
                break
        if flag:
            final_rect.append([x,y,w,h])
    rectangles = final_rect
    rectangles.sort()
    return rectangles

def gen_intermediate_file_name(filename: str, file_type: str, unique_identifier: str):
    return f"./intermediate_detection_files/{filename}_{unique_identifier}.{file_type}"

#fungsi ini dipanggil ketika sudah mendapatkan kotak plat nomor
def prepare_env_for_reading_license_plates(should_save_intermediate_files: bool):
    try:
        if should_save_intermediate_files:
            if os.path.exists("./intermediate_detection_files/"):
                shutil.rmtree("./intermediate_detection_files/")
            os.mkdir("./intermediate_detection_files/")
    except: pass

#baca satu kotak plat nomor
#mengembalikan plat nomor sebagai bentuk string
def read_license_plate(unique_identifier: str, box: any, original_image: Image, width_boost: int, additional_white_spacing_each_side: int, debug: bool, should_try_lp_crop: bool, minimum_number_of_chars_for_match: int) -> (Image, str): # type: ignore
    #crop gambar
    x_min, y_min, x_max, y_max = box.xyxy.cpu().detach().numpy()[0]
    original_width  = x_max - x_min
    original_height = y_max - y_min
    boost_multiplier = width_boost / original_width
    boosted_width = int(original_width * boost_multiplier)
    boosted_height = int(original_height * boost_multiplier)
    license_plate_crooped_img = original_image.crop(
        (x_min, y_min, x_max, y_max)
    ).convert(
        "L"
    ).resize(
        [boosted_width, boosted_height]
    )
    if should_try_lp_crop:
        license_plate_cropped_img = license_plate_cropped_img.crop(((45, 0, boosted_width - 20, boosted_height)))
    if debug:
        license_plate_crooped_img.save(gen_intermediate_file_name("cropped_license_plate_full", "jpg", unique_identifier))
    
    #pre-process image
    license_plate_crooped_img_as_np_array = cv2.cvtColor(np.array(license_plate_crooped_img), cv2.COLOR_GRAY2BGR)
    iwl_bb = clean_plate_into_contrours(license_plate_crooped_img_as_np_array, width_boost)
    iwl_wb = cv2.bitwise_not(iwl_bb)
    if debug:
        cv2.imwrite(gen_intermediate_file_name("iwl_bb", "jpg", unique_identifier), iwl_bb, [int(cv2.IMWRITE_JPEG_QUALITY),100])
        cv2.imwrite(gen_intermediate_file_name("iwl_wb", "jpg", unique_identifier), iwl_wb, [int(cv2.IMWRITE_JPEG_QUALITY),100])
    
    #get each letter
    letter_rectangles = get_letter_rectangles_from_contours(iwl_wb)
    iwl_wb_pil = Image.fromarray(iwl_wb, mode="L")

    if len(letter_rectangles) < minimum_number_of_chars_for_match:
        return(license_plate_cropped_img, "")

    def process_letter_rectangle(args):
        i, (x, y, w, h) = args
        letter_box_cropped_img = iwl_wb_pil.crop((x, y, x + w, y + h ))
        new_letter_box_img = Image.new(
            "L", (
                letter_box_cropped_img.width + (additional_white_spacing_each_side *2),
                letter_box_cropped_img.height + (additional_white_spacing_each_side *2)
            ), "white"
        )
        new_letter_box_img.paste(letter_box_cropped_img,(additional_white_spacing_each_side, additional_white_spacing_each_side))

        if debug:
            new_letter_box_img.save(gen_intermediate_file_name("cropped_image", "jpg", f"{unique_identifier}_{i}"))
        
        char_from_img = pytesseract.image_to_string(new_letter_box_img, lang='ind', config="--psm 8 --dpi 96 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        char_from_img = str(char_from_img).strip()

        if debug:
            print(f"{i}=> {char_from_img}")
        
        return char_from_img
    
    resulting_license_plate_string = ""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_letter_rectangle, enumerate(letter_rectangles)))
        resulting_license_plate_string = "".join(results)
    
    return (license_plate_crooped_img, resulting_license_plate_string.strip())

                       


