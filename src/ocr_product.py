from time import time
from io import BytesIO
import cv2
import numpy as np
import pytesseract

from PIL import Image

import src.auxiliary as aux


class OcrProduct:
    def __init__(self,
                 image,
                 language: str = 'eng',
                 spell_corrector=False,
                 show_performace: bool = False):
        """ # OcrProduct
        This class is responsible for the image processing and text extraction from product packaging images.

        Args:
            image (str, np.ndarray, PIL.Image): image to be processed.
            language (str, optional): language of the text to be extracted. Defaults to 'eng'.
            spell_corrector (bool, optional): if True, the text will be spell corrected. Defaults to False.
            show_performace (bool, optional): if True, the execution time will be shown. Defaults to False.

        Raises:
            TypeError: if language variable isn't a string, show_perf. and spell_corrector aren't bool.
            NotImplementedError: if the method to process the image isn't implemented yet.

        Returns:
            OcrProduct: object with the text extracted from the image.
        """
        self.define_global_vars(language, show_performace, spell_corrector)
        started_time = time()

        input_type = aux.get_input_type(image)
        self.text = self.process_image(image, input_type)

        if self.spell_corrector:
            sym_spell = aux.load_dict_to_memory()
            self.text = [aux.get_word_suggestion(
                sym_spell, input_term) for input_term in self.text.split(' ')]
            self.text = ' '.join(self.text)

        self.execution_time = time() - started_time

    def __repr__(self):
        return repr(self.text) \
            if not self.show_performace \
            else repr([self.text, self.show_performace])

    def define_global_vars(self, language: str, show_performace: bool, spell_corrector: bool) -> None:
        """ # Define Global Variables
        This method defines the global variables of the class.

        Args:
            language (str): The language of the text to be extracted.
            show_performace (bool): If True, the execution time will be shown.
            spell_corrector (bool): If True, the text will be spell corrected.

        Raises:
            TypeError: if language variable isn't a string, show_perf. and spell_corrector aren't bool.
        """
        if isinstance(language, str) and isinstance(show_performace, bool) \
                and isinstance(spell_corrector, bool):
            self.lang = language
            self.show_performace = show_performace
            self.spell_corrector = spell_corrector
        else:
            raise TypeError(
                'language variable must be a string, show_perf. and spell_corrector bool!')

    def process_image(self, image, _type: int) -> str:
        """ # Process Image
        This method is responsible for processing the image and extracting the text from it.

        Args:
            image (str, np.ndarray, PIL.Image): image to be processed.
            _type (int): type of the input image.

        Raises:
            NotImplementedError: if the method to process the image isn't implemented yet.

        Returns:
            str: text extracted from the image.
        """
        if _type == 1:
            processed_img = self.run_online_img_ocr(image)
        elif _type == 2:
            processed_img = self.run_path_img_ocr(image)
        elif _type == 3:
            processed_img = self.run_img_ocr(image)
        else:
            raise NotImplementedError(
                'method to this specific processing isn'"'"'t implemented yet!')
        return processed_img

    def run_online_img_ocr(self, image_url: str) -> str:
        """ # Run Online Image OCR
        This method is responsible for processing the image and extracting the text from it.

        Args:
            image_url (str): url of the image to be processed.

        Returns:
            str: text extracted from the image.
        """
        image = aux.get_image_from_url(image_url)
        phrase = read_text(Image.open(BytesIO(image.content)), self.lang)
        return phrase

    def run_path_img_ocr(self, image: str) -> str:
        """ # Run Path Image OCR
        This method is responsible for processing the image and extracting the text from it.

        Args:
            image (str): path of the image to be processed.

        Returns:
            str: text extracted from the image.
        """
        phrase = read_text(Image.open(image), self.lang)
        return phrase

    def run_img_ocr(self, image) -> str:
        """ # Run Image OCR
        This method is responsible for processing the image and extracting the text from it.

        Args:
            image (Image, np.ndarray): image to be processed.

        Returns:
            str: text extracted from the image.
        """
        phrase = read_text(image, self.lang)
        return phrase


def read_text(image, language: str) -> str:
    variants = build_image_variants(image)
    configs = [
        "--oem 3 --psm 6",
        "--oem 3 --psm 11",
        "--oem 3 --psm 4",
        "--oem 3 --psm 3",
    ]

    best_text = ""
    best_score = -1.0

    for variant in variants:
        for config in configs:
            text, score = read_text_with_confidence(variant, language, config)
            if score > best_score and text.strip():
                best_score = score
                best_text = text

    return best_text


def build_image_variants(image) -> list:
    if isinstance(image, np.ndarray):
        cv_image = image.copy()
    else:
        cv_image = np.asarray(image.convert("RGB"))[:, :, ::-1]

    cv_image = aux.remove_alpha_channel(cv_image)
    height, width = cv_image.shape[:2]
    scale = max(1, int(1600 / max(height, width)))
    if scale > 1:
        cv_image = cv2.resize(
            cv_image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    equalized = cv2.equalizeHist(gray)
    adaptive = cv2.adaptiveThreshold(
        equalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    otsu = cv2.threshold(
        equalized,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )[1]

    return [cv_image, gray, equalized, adaptive, otsu]


def read_text_with_confidence(image, language: str, config: str) -> tuple:
    data = pytesseract.image_to_data(
        image,
        lang=language,
        config=config,
        output_type=pytesseract.Output.DICT,
    )

    words = []
    confidences = []
    for word, confidence in zip(data["text"], data["conf"]):
        word = word.strip()
        if not word:
            continue

        try:
            confidence_value = float(confidence)
        except ValueError:
            continue

        if confidence_value >= 0:
            words.append(word)
            confidences.append(confidence_value)

    if not words:
        return "", -1.0

    text = " ".join(words)
    score = sum(confidences) / len(confidences)
    return text, score
