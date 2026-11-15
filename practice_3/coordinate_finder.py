import sys
from PIL import Image
import matplotlib.pyplot as plt


def show_coordinates(image_path):
    img = Image.open(image_path).convert("RGB")

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(f"Click to get coordinates: {image_path}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    def on_click(event):
        if event.inaxes == ax:
            x = int(round(event.xdata))
            y = int(round(event.ydata))
            print(f"x={x}, y={y}")

    fig.canvas.mpl_connect("button_press_event", on_click)

    plt.show()

if len(sys.argv) != 2:
    print("Usage: python coordinate_finder.py path/to/template.jpg")
else:
    show_coordinates(sys.argv[1])