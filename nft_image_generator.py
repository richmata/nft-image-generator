import cv2
import click
import numpy as np
import pandas as pd
from PIL import Image


class ImageGenerator:
    DIR = "images"
    FILE_PATH = f"{DIR}/new_images/new_image.png"

    def __init__(self, frame: str, body: str, head: str):
        self.frame = frame
        self.body = body
        self.head = head
        self.image = cv2.imread(self.FILE_PATH)
        

    @property
    def generate_new_image(self) -> np.array:
        """
        Layers existing images to create new image
        and returns as grayscale
        """
        new_image = Image.new("RGB", (1200, 1200), (255, 255, 255)) # blank canvas
        
        # layer frame image
        # TODO: categorize contours by shape/size and colorize
        frame = Image.open(self.frame)
        new_image.paste(frame, (0, 0), frame)
        
        # layer body image
        # TODO: categorize contours by shape/size and colorize
        body = Image.open(self.body)
        new_image.paste(body, (0, 0), body)
        
        # layer head image
        # TODO: categorize contours by shape/size and colorize
        head = Image.open(self.head)
        new_image.paste(head, (0, 0), head)

        # TODO: make a call to save image in database
        # image.save(self.FILE_PATH)

        # convert from PIL.Image to openCV format
        cv2_image = np.array(new_image)
        cv2_image = cv2_image[:, :, ::-1].copy() # RGB to BGR conversion

        cv2.imwrite(f"{self.DIR}/new_images/new_image.png", cv2_image)
        
        return cv2_image
    
    @staticmethod
    def _find_contours(grayscale):
        """
        Take the grayscale of a new image and returns the contours

        Parameters
        ----------
        grayscale: openCV grayscale of an image
        """
        _, binary = cv2.threshold(grayscale, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    @staticmethod
    def categorize_contour(contour) -> tuple:
        """
        Categorizes contour by shape and size
        """

        # find approximate polygon and polygon area
        polygon = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)            
        polygon_area = cv2.contourArea(contour)
        return (polygon, polygon_area)

    @staticmethod
    def _find_center_point(contour) -> tuple:
        M = cv2.moments(contour)
        if M["m00"] != 0.0:
            x = int(M["m10"]/M["m00"])
            y = int(M["m01"]/M["m00"])
        return (x, y)

    @property
    def generate_nfts(self):
        """
        1. get grayscale of a new image and convert to binary to find contours
        3. find center point of each contour and draw a circle with radius 5px
        4. calculate mean RGB value of pixels within each circle
        5. identify contours with mean RGB value being white
        6. categorize white contours by shape/size and colorfill
        7. save image to database
        """

        # TODO: this will be a helper function for looping
        # through all image combinations in a database

        data = {
            "image": self.generate_new_image, 
            "grayscale": None, 
            "contours": None, 
            "average_contour_color": None, 
            "image_whitespace": None
        }

        image_df = pd.DataFrame(data=data)
        image_df["grayscale"] = image_df.apply(lambda x: cv2.cvtColor(x["image"], cv2.COLOR_BGR2GRAY), axis=1)
        image_df["contours"] = image_df["grayscale"].apply(self._find_contours)
        image_df["average_contour_color"] = image_df.apply(lambda x: np.average(x["image"], axis=0), axis=1)
        image_df["image_whitespace"] = (
            image_df["average_contour_color"].apply(lambda x: [color for color in x if color == 255], axis=1)
        )
        # for contour in contours:

        #     # find center point of contour
        #     center = self._find_center_point(contour)

        #     # draw circles with radius 5px in each contour
        #     # and identify mean color value of the bounded pixels
        #     image = cv2.circle(image, center, 10, (0), 1)

            
        #     # use mask to find mean color values within the circles
        #     # means = cv2.mean(image, mask=binary)

        #     # fill the contour based on shape type and size
        #     # cv2.fillPoly(image, pts=[contour], color=(255,0,0)) # black
        
        return cv2.imwrite(f"{self.DIR}/new_images/nft_image.png", image_df["whitespace"])

    
    
    # @property
    # def largest_shape(self):
    #     image = self.image
        
    #     areas = []
    #     for contour in self.image_contours:
    #         ar = cv2.contourArea(contour)
    #         areas.append(ar)

    #     max_area = min(areas)
    #     max_area_index = areas.index(max_area) # index of the list element with largest area

    #     cnt = self.image_contours[max_area_index] # largest area
        
    #     cv2.drawContours(image, [cnt], 0, (255, 0, 0), 5)
    #     _, ax = plt.subplots(figsize=(20, 20))
    #     return ax.imshow(image)


# @click.command()
# @click.option("--frame", default="images/frames/Frame_3.png", help="File path for frame image")
# @click.option("--body", default="images/bodies/Gator_Body.png", help="File path for body image")
# @click.option("--head", default="images/heads/Punk_Head.png", help="File path for head image")
# def generate_nft(frame, body, head):
#     NFTImageGenerator(frame, body, head).generate_nfts


# if __name__ == "__main__":
#     generate_nft()


# TODO: 
    # identify frame
    # identify body shapes and categorize (if shapes are similar group together for color filling)
    # identify head shapes and categorize (if shapes are similar group together for color filling)