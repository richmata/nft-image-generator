import cv2
import click
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


class NFTImageGenerator:
    DIR = "images"
    FILE_PATH = f"{DIR}/new_images/new_image.png"

    def __init__(self, frame: str, body: str, head: str):
        self.frame = frame
        self.body = body
        self.head = head
        self.image = cv2.imread(self.FILE_PATH)
        self.grayscale = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
    
    @property
    def image_contours(self):
        """
        Returns contours found in an image
        """
        # df = pd.DataFrame([], columns=["contour", "poly", "shape", "area", "center"])
        _, binary = cv2.threshold(self.grayscale, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    @property
    def generate_nft_image(self):
        """
        1. get grayscale of image and convert to binary
        2. find contours
        3. get center point of each contour and draw a circle with radius 5 px
        4. get mean RGB value of pixels within the circle
        5. identify contours with mean RGB value being white ()
        6. categorize white contours by shape and color fill
        7. update image
        """
        image = self.generate_new_image
        for contour in self.image_contours:

            # find approximate polygon and polygon area
            # poly = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)            
            # area = cv2.contourArea(contour)

            # find center point of contour
            M = cv2.moments(contour)
            if M["m00"] != 0.0:
                x = int(M["m10"]/M["m00"])
                y = int(M["m01"]/M["m00"])

            # draw circles with radius 5px in each contour
            # and identify mean color value of the bounded pixels
            image = cv2.circle(image, (x, y), 10, (0), 1)
            mask = np.array([255, 255, 255])
            # use mask to find mean color values within the circles
            # means = cv2.mean(image, mask=binary)

            # fill the contour based on shape type and size
            # cv2.fillPoly(image, pts=[contour], color=(255,0,0)) # black
            
        # return categories
        return cv2.imwrite(f"{self.DIR}/new_images/colorized.png", image)

    # @property
    # def colorized(self):
    #     """
    #     Colors images generated using Images.generate
    #     """
    #     image = self.image
    #     gray = self.grayscale
    #     mask = np.zeros_like(gray)

    #     # loop through contours and categorize polygons
    #     for contour in self.contours:
  
    #         # find approximate polygon and polygon area
    #         poly = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

    #         # # # create mask for polygon
    #         cv2.fillPoly(image, [contour], (0,255,0))
    #         # cv2.drawContours(image, [poly], 0, (255, 0, 0), 5)

    #         # # get color values in gray image corresponding to where mask is white
    #         # values = gray[np.where(mask == 255)]
            
    #         # draw contours on image
    #         cv2.drawContours(image, [contour], 0, (255, 0, 0), 5)
  
    #         if len(poly) == 4:
    #             cv2.fillPoly(image, pts=[contour], color=(0,0,0)) # black

    #     return cv2.imwrite(f"{self.DIR}/new_images/colorized.png", image)

    @property
    def generate_new_image(self):
        """
        Layers existing images and save as a new image
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
        
        return 
    
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


@click.command()
@click.option("--frame", default="images/frames/Frame_3.png", help="File path for frame image")
@click.option("--body", default="images/bodies/Gator_Body.png", help="File path for body image")
@click.option("--head", default="images/heads/Punk_Head.png", help="File path for head image")
def generate_nft(frame, body, head):
    NFT = NFTImageGenerator(frame, body, head)
    NFT.generate_new_image
    NFT.colorize_image


if __name__ == "__main__":
    generate_nft()


# TODO: 
    # identify frame
    # identify body shapes and categorize (if shapes are similar group together for color filling)
    # identify head shapes and categorize (if shapes are similar group together for color filling)