from PIL import Image, ImageDraw, ImageFont


def get_text_bbox(input_text, font):
    img = Image.new("RGB", (1, 1))
    img = ImageDraw.Draw(img, "RGB")
    return img.textbbox((0, 0), input_text, font)


def get_text_size(input_text, font):
    img = Image.new("RGB", (1, 1))
    img = ImageDraw.Draw(img, "RGB")
    x1, y1, x2, y2 = img.textbbox((0, 0), input_text, font)
    """
    Return: (width, height), baseline
    """
    return (x2 - x1, y2 - y1), y1


def get_text_length(input_text, font):
    img = Image.new("RGB", (1, 1))
    img = ImageDraw.Draw(img, "RGB")
    return img.textlength(input_text, font)


def draw_text(text: str):
    position = 1, 1
    font_name = "Arial.ttf"
    font_size = 18
    font = ImageFont.truetype(font_name, font_size)
    text_color = (255, 255, 255)
    bg_color = (0, 0, 0)

    x1, y1, x2, y2 = get_text_bbox(text, font)
    img = Image.new("RGB", (1, 1), bg_color)
    drawer = ImageDraw.Draw(img)
    drawer.text(position, text, text_color, font=font)
    img.save("img.png")
