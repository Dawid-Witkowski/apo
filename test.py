import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import numpy as np
from PIL import Image, ImageTk
import statistics
import cv2
import xlsxwriter


class ImageViewer:
    def __init__(self, root, img=None):
        self.root = root
        self.root.title("Image Viewer")
        self.images = []
        self.isGrayscale = []
        self.current_image_index = 0
        self.canvas = tk.Canvas(self.root)
        self.canvas.pack(fill="both", expand=True)

        self.menu = tk.Menu(self.root)
        self.root.config(menu=self.menu)

        file_menu = tk.Menu(self.menu)
        self.menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Image", command=self.load_image)
        file_menu.add_command(label="Save Image", command=self.save_image)
        file_menu.add_command(label="Duplicate Image", command=self.duplicate_image)
        file_menu.add_command(label="Exit", command=self.root.destroy)
        file_menu.add_command(label="Generate LUT table", command=self.generate_lut)

        resize_menu = tk.Menu(self.menu)
        self.menu.add_cascade(label="Resize", menu=resize_menu)
        resize_menu.add_command(label="Original Size", command=self.show_original_size)
        resize_menu.add_command(label="Full Screen", command=self.full_screen)
        resize_menu.add_command(label="Window Size", command=self.window_size)

        histogram_menu = tk.Menu(self.menu)
        self.menu.add_cascade(label="Histogram", menu=histogram_menu)
        histogram_menu.add_command(label="Show histogram", command=self.show_histogram)

        hist_transform_menu = tk.Menu(self.menu)
        self.menu.add_cascade(label="Transformations", menu=hist_transform_menu)
        hist_transform_menu.add_command(label="Linear stretch", command=lambda: self.linear_stretch(False))
        hist_transform_menu.add_command(label="Linear stretch (5% wysycenie)",
                                        command=lambda: self.linear_stretch(True))

        hist_transform_menu.add_command(label="Histogram equalization", command=self.histogram_equalization)
        hist_transform_menu.add_command(label="binary tresh", command=self.binary_threshold)
        hist_transform_menu.add_command(label="binary tresh gray", command=self.gray_threshold)

        point_ops_menu = tk.Menu(self.menu)
        self.menu.add_cascade(label="Point Operations", menu=point_ops_menu)
        point_ops_menu.add_command(label="Negation", command=self.negation)
        point_ops_menu.add_command(label="Gray level reduction", command=self.gray_level_reduction)

        point_ops_menu.add_command(label="Multi-Add (Max 5)",
                                   command=lambda: self.setup_multi_add())

        point_ops_menu.add_command(label="Scalar Add (+V)", command=lambda: self.setup_scalar_operation('add'))
        point_ops_menu.add_command(label="Scalar Multiply (*V)",
                                   command=lambda: self.setup_scalar_operation('multiply'))
        point_ops_menu.add_command(label="Scalar Divide (/V)", command=lambda: self.setup_scalar_operation('divide'))
        point_ops_menu.add_command(label="Absolute Difference", command=self.setup_absolute_difference)

        cv_menu = tk.Menu(self.menu)
        self.menu.add_cascade(label="Cv2", menu=cv_menu)
        cv_menu.add_command(label="average", command=lambda: self.setup_neighbor_operation("average"))
        cv_menu.add_command(label="weighted average",
                            command=lambda: self.setup_neighbor_operation("weighted"))
        cv_menu.add_command(label="gaussian", command=lambda: self.setup_neighbor_operation("gaussowski"))
        cv_menu.add_command(label="Sharpening", command=self.setup_linear_sharpening)
        cv_menu.add_command(label="Prewitt", command=self.setup_prewitt_detection)
        cv_menu.add_command(label="sobel", command=self.setup_sobel_detection)
        cv_menu.add_command(label="Canny", command=self.setup_canny_detection)
        cv_menu.add_command(label="Median", command=self.setup_median_filter)

        logic_menu = tk.Menu(self.menu)
        self.menu.add_cascade(label="Logic", menu=logic_menu)
        logic_menu.add_command(label="NOT", command=self.perform_not)
        logic_menu.add_command(label="AND", command=self.perform_and)
        logic_menu.add_command(label="OR", command=self.perform_or)
        logic_menu.add_command(label="XOR", command=self.perform_xor)

        lab3 = tk.Menu(self.menu)
        self.menu.add_cascade(label="Lab3", menu=lab3)
        lab3.add_command(label="Histogram streeeetch", command=self.setup_custom_stretch)
        lab3.add_command(label="2 value segmenting", command=self.setup_two_value_segment)
        lab3.add_command(label="otsu", command=self.setup_otsu_thresholding)
        lab3.add_command(label="adaptive threshold", command=self.setup_adaptive_threshold)
        lab3.add_command(label="morph", command=self.setup_morphology_operation)
        lab3.add_command(label="skeletonize", command=self.setup_skeletonization)

        lab4 = tk.Menu(self.menu)
        self.menu.add_cascade(label="lab4", menu=lab4)
        lab4.add_command(label="moments", command=self.calculate_moments)
        lab4.add_command(label="Hought", command=self.hough_edge_detection)

        if img:
            self.images.append(img.copy())
            self.current_image_index = 0
            self.display_image()


    def grayscale_narrowing(self, image, low: int, high: int) -> np.ndarray:
        arr = np.array(image)
        narrowed = (arr * ((high - low) / 255) + low).astype('uint8')
        return narrowed

    def load_image(self):
        filepath = filedialog.askopenfilename(filetypes=[("Images", "*.bmp *.tif *.png *.jpg")])
        if filepath:
            try:
                img = Image.open(filepath).convert('RGB')
                w, h = img.size
                isGrayscale = True
                for i in range(w):
                    for j in range(h):
                        r, g, b = img.getpixel((i, j))
                        if r != g or g != b:
                            isGrayscale = False
                            break
                    if not isGrayscale:
                        break
                self.isGrayscale.append(isGrayscale)

                self.images.append(img)
                self.current_image_index = len(self.images) - 1
                self.display_image()

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")

    def save_image(self):
        if not self.images:
            messagebox.showwarning("No Image", "No image to save")
            return
        img = self.images[self.current_image_index]
        filepath = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Files", "*.png")])
        if filepath:
            try:
                img.save(filepath)
                messagebox.showinfo("Success", "Image saved successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {e}")

    def generate_lut(self):
        if not self.images:
            messagebox.showwarning("No Image", "No image to generate LUT")
            return
        img = self.images[self.current_image_index]
        is_grayscale = self.isGrayscale[self.current_image_index]
        img = img.convert("RGB")
        red = [0] * 256
        green = [0] * 256
        blue = [0] * 256
        width, height = img.size
        for x in range(width):
            for y in range(height):
                r, g, b = img.getpixel((x, y))
                red[r] += 1
                green[g] += 1
                blue[b] += 1
        lut_window = tk.Toplevel(self.root)
        lut_window.title("LUT Table")
        text = tk.Text(lut_window, wrap="none")
        text.pack(fill="both", expand=True)
        if is_grayscale:
            text.insert("end", f"{'Value':<8} {'Gray':<10}\n")
            for i in range(256):
                gray = int((red[i] + green[i] + blue[i]) / 3)
                text.insert("end", f"{i:<8} {gray:<10}\n")
        else:
            text.insert("end", f"{'Value':<8} {'Red':<10} {'Green':<10} {'Blue':<10}\n")
            for i in range(256):
                text.insert("end", f"{i:<8} {red[i]:<10} {green[i]:<10} {blue[i]:<10}\n")

    def duplicate_image(self):
        if not self.images:
            messagebox.showwarning("No Image", "No image to duplicate")
            return
        img = self.images[self.current_image_index]
        new_window = tk.Toplevel()
        ImageViewer(new_window, img)

    def display_image(self):
        if not self.images:
            return
        img = self.images[self.current_image_index]
        img_tk = ImageTk.PhotoImage(img)
        self.root.geometry(f"{img.width}x{img.height}")
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=img_tk, anchor=tk.NW)
        self.canvas.image = img_tk

    def show_original_size(self):
        if not self.images:
            return
        img = self.images[self.current_image_index]
        img_tk = ImageTk.PhotoImage(img)
        self.root.attributes("-fullscreen", False)
        self.root.geometry(f"{img.width}x{img.height}")
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
        self.canvas.create_image(0, 0, image=img_tk, anchor=tk.NW)
        self.canvas.image = img_tk

    def full_screen(self):
        if not self.images:
            return
        self.root.attributes("-fullscreen", True)
        img = self.images[self.current_image_index]
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        img_ratio = img.width / img.height
        screen_ratio = screen_width / screen_height
        if img_ratio > screen_ratio:
            new_width = screen_width
            new_height = int(screen_width / img_ratio)
        else:
            new_height = screen_height
            new_width = int(screen_height * img_ratio)
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(resized_img)
        self.canvas.delete("all")
        self.canvas.config(width=screen_width, height=screen_height, bg="black")
        self.canvas.create_image((screen_width - new_width) // 2, (screen_height - new_height) // 2, image=img_tk,
                                 anchor=tk.NW)
        self.canvas.image = img_tk

    def window_size(self):
        if not self.images:
            return
        self.root.attributes("-fullscreen", False)
        self.root.update_idletasks()
        window_width = self.root.winfo_width()
        window_height = self.root.winfo_height()
        img = self.images[self.current_image_index]
        img_width, img_height = img.size
        img_ratio = img_width / img_height
        window_ratio = window_width / window_height
        if img_ratio > window_ratio:
            new_width = window_width
            new_height = int(window_width / img_ratio)
        else:
            new_height = window_height
            new_width = int(window_height * img_ratio)
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(resized_img)
        self.canvas.delete("all")
        self.canvas.config(width=window_width, height=window_height)
        self.canvas.create_image((window_width - new_width) // 2, (window_height - new_height) // 2, image=img_tk,
                                 anchor=tk.NW)
        self.canvas.image = img_tk

    def show_histogram(self):
        if not self.images:
            messagebox.showwarning("No Image", "No image to analyze")
            return

        img = self.images[self.current_image_index]
        width, height = img.size

        if self.isGrayscale[self.current_image_index]:
            img = img.convert("L")
            histogram = [0] * 256
            pixels = []

            for x in range(width):
                for y in range(height):
                    val = img.getpixel((x, y))
                    histogram[val] += 1
                    pixels.append(val)

            min_val = min(pixels)
            max_val = max(pixels)
            mean_val = statistics.mean(pixels)
            median_val = statistics.median(pixels)
            mode_val = statistics.mode(pixels)
            std_val = statistics.stdev(pixels)

            hist_window = tk.Toplevel(self.root)
            hist_window.title("Grayscale Histogram")
            hist_window.geometry("700x500")

            stats_label = tk.Label(hist_window, text=(
                f"Min: {min_val}   Max: {max_val}\n"
                f"Mean: {mean_val:.2f}   Median: {median_val}\n"
                f"Most popular value/mode: {mode_val}   Std Dev: {std_val:.2f}\n"
                f"Number of pixels: {height * width}"
            ), justify="left", anchor="w")
            stats_label.pack(fill="x", padx=10, pady=5)

            canvas_width = 650
            canvas_height = 400
            left_padding = 50
            bottom_padding = 30
            top_padding = 10

            canvas = tk.Canvas(hist_window, width=canvas_width, height=canvas_height, bg="white")
            canvas.pack(padx=10, pady=10)

            max_h = max(histogram)
            bar_width = 2
            usable_height = canvas_height - bottom_padding - top_padding
            scale_y = usable_height / max_h

            for i in range(256):
                h = histogram[i] * scale_y
                x0 = left_padding + i * bar_width
                y0 = canvas_height - bottom_padding
                x1 = x0 + bar_width
                y1 = y0 - h
                canvas.create_rectangle(x0, y0, x1, y1, fill="black", outline="")

            canvas.create_line(left_padding, canvas_height - bottom_padding, canvas_width,
                               canvas_height - bottom_padding,
                               fill="black")

            for i in range(0, 256, 50):
                x = left_padding + i * bar_width
                canvas.create_text(x, canvas_height - bottom_padding + 15, text=str(i), anchor="n")

            canvas.create_line(left_padding, top_padding, left_padding, canvas_height - bottom_padding, fill="black")

            num_ticks = 5
            step = max_h // num_ticks if max_h >= num_ticks else 1
            for t in range(0, max_h + 1, step):
                y = canvas_height - bottom_padding - t * scale_y
                canvas.create_line(left_padding - 5, y, left_padding, y, fill="black")
                canvas.create_text(left_padding - 10, y, text=str(t), anchor="e")

        else:
            red_histogram = [0] * 256
            green_histogram = [0] * 256
            blue_histogram = [0] * 256

            for x in range(width):
                for y in range(height):
                    r, g, b = img.getpixel((x, y))
                    red_histogram[r] += 1
                    green_histogram[g] += 1
                    blue_histogram[b] += 1

            colors = [
                ("Red Channel", red_histogram, "red"),
                ("Green Channel", green_histogram, "green"),
                ("Blue Channel", blue_histogram, "blue")
            ]

            for title, histogram, color in colors:
                channel_window = tk.Toplevel(self.root)
                channel_window.title(title)
                channel_window.geometry("600x450")

                canvas = tk.Canvas(channel_window, width=550, height=350, bg="white")
                canvas.pack(padx=10, pady=10)

                max_h = max(histogram)
                bar_width = 2
                left_padding = 50
                bottom_padding = 30
                top_padding = 10
                usable_height = 350 - bottom_padding - top_padding
                scale_y = usable_height / max_h

                min_val = min(histogram)
                max_val = max(histogram)
                mean_val = statistics.mean(histogram)
                median_val = statistics.median(histogram)
                mode_val = statistics.mode(histogram)
                std_val = statistics.stdev(histogram)

                stats_label = tk.Label(channel_window, text=(
                    f"Min: {min_val}   Max: {max_val}\n"
                    f"Mean: {mean_val:.2f}   Median: {median_val}\n"
                    f"Most popular value/mode: {mode_val}   Std Dev: {std_val:.2f}\n"
                    f"Number of pixels: {height * width}"
                ), justify="left", anchor="w")
                stats_label.pack(fill="x", padx=10, pady=5)

                for i in range(256):
                    h = histogram[i] * scale_y
                    x0 = left_padding + i * bar_width
                    y0 = 350 - bottom_padding
                    x1 = x0 + bar_width
                    y1 = y0 - h
                    canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="")

                canvas.create_line(left_padding, 350 - bottom_padding, 550,
                                   350 - bottom_padding,
                                   fill="black")

                for i in range(0, 256, 50):
                    x = left_padding + i * bar_width
                    canvas.create_text(x, 350 - bottom_padding + 15, text=str(i), anchor="n")

                canvas.create_line(left_padding, top_padding, left_padding,
                                   350 - bottom_padding, fill="black")

                num_ticks = 5
                step = max_h // num_ticks if max_h >= num_ticks else 1
                for t in range(0, max_h + 1, step):
                    y = 350 - bottom_padding - t * scale_y
                    canvas.create_line(left_padding - 5, y, left_padding, y, fill="black")
                    canvas.create_text(left_padding - 10, y, text=str(t), anchor="e")

    def negation(self):
        if not self.images:
            messagebox.showwarning("No Image", "No image to process")
            return
        img = self.images[self.current_image_index].convert("L")
        neg_pixels = [255 - v for v in img.getdata()]
        new_img = Image.new("L", img.size)
        new_img.putdata(neg_pixels)
        self.images.append(new_img.convert("RGB"))
        self.isGrayscale.append(True)
        self.current_image_index = len(self.images) - 1
        self.display_image()

    def gray_level_reduction(self):
        if not self.images:
            messagebox.showwarning("No Image", "No image to process")
            return

        levels_str = simpledialog.askstring("Levels", "2-256:")
        if not levels_str:
            return

        try:
            levels = int(levels_str)
            if levels < 2 or levels > 256:
                messagebox.showerror("Error", "2-256")
                return
        except ValueError:
            messagebox.showerror("Error", "wrong value")
            return

        img = self.images[self.current_image_index].convert("L")
        original_pixels = list(img.getdata())
        new_pixels = []

        # jak chcemy zmniejszyć liczbę szarości do 4 to 255 / 3 = 85
        scale_factor = 255 / (levels - 1)

        for pixel_value in original_pixels:
            # obliczenie do jakiej "grupy" szarości należy pixel
            level_index = (pixel_value * levels) // 256

            # przywracam do 0-255
            new_value = int(level_index * scale_factor)

            new_pixels.append(new_value)

        new_img = Image.new("L", img.size)
        new_img.putdata(new_pixels)

        self.images.append(new_img.convert("RGB"))
        self.isGrayscale.append(True)
        self.current_image_index = len(self.images) - 1
        self.display_image()

    def linear_stretch(self, saturate=False):
        if not self.images:
            messagebox.showwarning("No Image", "No image to process")
            return
        img = self.images[self.current_image_index].convert("L")
        pixels = list(img.getdata())
        sorted_pixels = sorted(pixels)
        if saturate:
            cut = int(0.05 * len(sorted_pixels))
            min_val = sorted_pixels[cut]
            max_val = sorted_pixels[-cut - 1]
        else:
            min_val = min(sorted_pixels)
            max_val = max(sorted_pixels)

        def stretch(v):
            return max(0, min(255, int((v - min_val) * 255 / (max_val - min_val))))

        stretched_pixels = [stretch(v) for v in pixels]
        new_img = Image.new("L", img.size)
        new_img.putdata(stretched_pixels)
        self.images.append(new_img.convert("RGB"))
        self.isGrayscale.append(True)
        self.current_image_index = len(self.images) - 1
        self.display_image()

    def histogram_equalization(self):
        if not self.images:
            messagebox.showwarning("No Image", "No image to process")
            return

        img = self.images[self.current_image_index].convert("L")
        pixels = list(img.getdata())

        hist = [0] * 256
        for p in pixels:
            hist[p] += 1

        cdf = np.cumsum(hist)

        cdf_min = cdf[cdf > 0].min()
        total_pixels = len(pixels)

        lut = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            if cdf[i] > 0:
                lut[i] = np.round(((cdf[i] - cdf_min) / (total_pixels - cdf_min)) * 255)
            else:
                lut[i] = 0

        result = [int(lut[p]) for p in pixels]

        new_img = Image.new("L", img.size)
        new_img.putdata(result)

        self.images.append(new_img)
        self.isGrayscale.append(True)
        self.current_image_index = len(self.images) - 1
        self.display_image()

    def binary_threshold(self):
        if not self.images:
            messagebox.showwarning("No Image", "No image to process")
            return

        threshold_str = simpledialog.askstring("Threshold", "0-255:")
        if not threshold_str:
            return

        try:
            threshold = int(threshold_str)
            if threshold < 0 or threshold > 255:
                messagebox.showerror("Error", "0-255")
                return
        except ValueError:
            messagebox.showerror("Error", "wrong values")
            return

        img = self.images[self.current_image_index].convert("L")
        pixels = list(img.getdata())

        new_pixels = [255 if v >= threshold else 0 for v in pixels]

        new_img = Image.new("L", img.size)
        new_img.putdata(new_pixels)
        self.images.append(new_img.convert("RGB"))
        self.isGrayscale.append(True)
        self.current_image_index = len(self.images) - 1
        self.display_image()

    def gray_threshold(self):
        if not self.images:
            messagebox.showwarning("No Image", "No image to process")
            return

        threshold_str = simpledialog.askstring("Threshold", "0-255:")
        if not threshold_str:
            return

        try:
            threshold = int(threshold_str)
            if threshold < 0 or threshold > 255:
                messagebox.showerror("Error", "0-255")
                return
        except ValueError:
            messagebox.showerror("Error", "Invalid threshold value")
            return

        img = self.images[self.current_image_index].convert("L")
        img_array = np.array(img)
        new_array = np.where(img_array >= threshold, img_array, 0).astype(np.uint8)
        new_img = Image.fromarray(new_array, mode="L")

        self.images.append(new_img.convert("RGB"))
        self.isGrayscale.append(True)
        self.current_image_index = len(self.images) - 1
        self.display_image()

    def setup_multi_add(self):
        images_to_process = []

        wysycenie = messagebox.askyesno(
            "wysycenie",
            "with wysycenie?"
        )

        # if self.images:
        #     current = self.images[self.current_image_index]
        #     images_to_process.append(current.convert("L"))

        slots_left = 5 - len(images_to_process)
        if slots_left <= 0:
            messagebox.showwarning("Limit Reached", "You already have 5 or more images loaded/selected.")
            return

        filepaths = filedialog.askopenfilenames(
            title=f"Select up to {slots_left} additional images",
            filetypes=[("Images", "*.bmp *.tif *.png *.jpg")]
        )

        if not filepaths:
            if len(images_to_process) < 2:
                return
        else:
            if len(filepaths) > slots_left:
                messagebox.showwarning("Limit Exceeded", f"You selected too many files. Taking first {slots_left}.")
                filepaths = filepaths[:slots_left]

            base_size = images_to_process[0].size if images_to_process else None

            for fp in filepaths:
                try:
                    img = Image.open(fp).convert("L")
                    if base_size is None:
                        base_size = img.size

                    if img.size != base_size:
                        messagebox.showerror("Size Error", f"Image {fp} has different size than others. Skipping.")
                        continue

                    images_to_process.append(img)
                except Exception as e:
                    messagebox.showerror("Error", f"Could not load {fp}: {e}")

        if len(images_to_process) < 2:
            messagebox.showwarning("Not enough images", "Need at least 2 images to perform addition.")
            return

        self.calculate_multi_add(images_to_process, wysycenie)

    # todo: check what is "wysycenie" in english XDD
    def calculate_multi_add(self, images, wysycenie=True):
        num_images = len(images)
        if num_images < 2:
            messagebox.showwarning("", "You need at least 2 images to combine.")
        if num_images > 5:
            messagebox.showwarning("at most 5 images.")

        np_images = []
        for img in images:
            gray = img.convert("L")
            arr = np.array(gray, dtype=np.uint8)
            np_images.append(arr)

        base_shape = np_images[0].shape
        for i, arr in enumerate(np_images[1:], start=2):
            if arr.shape != base_shape:
                raise ValueError(
                    f"Image {i} has different size: {arr.shape[1]}x{arr.shape[0]} "
                    f"(expected: {base_shape[1]}x{base_shape[0]})"
                )

        result = np.zeros_like(np_images[0], dtype=np.float32)
        if wysycenie:
            for arr in np_images:
                result += arr.astype(np.float32)
        else:
            max_scale_value = 255 // num_images
            scaled_images = []
            for arr in np_images:
                scaled = cv2.normalize(arr, None, 0, max_scale_value, cv2.NORM_MINMAX)
                scaled_images.append(scaled)
            for arr in scaled_images:
                result += arr.astype(np.float32)

        result = np.clip(result, 0, 255).astype(np.uint8)
        result_img = Image.fromarray(result, mode="L")

        self.images.append(result_img.convert("RGB"))
        self.isGrayscale.append(True)
        self.current_image_index = len(self.images) - 1
        self.display_image()

        mode_str = "with wysycenie" if wysycenie else "without wysycenie (scaled)"
        messagebox.showinfo("Success", f"Combined {num_images} images {mode_str}.")

    def setup_scalar_operation(self, operation):
        if not self.images:
            messagebox.showwarning("Load in picture", "First load in an image.")
            return

        img = self.images[self.current_image_index].convert("L")

        value_str = simpledialog.askstring("value", operation)
        if not value_str:
            return

        try:
            value = abs(int(value_str))
        except ValueError:
            return

        wysycenie = messagebox.askyesno(
            "wysycenie",
            "with wysycenie?"
        )

        self.perform_scalar_operation(img, operation, value, wysycenie)

    def perform_scalar_operation(self, img, operation, value, wysycenie):
        pixels = list(img.getdata())
        result_pixels = []

        # bruh
        if operation == 'divide' and value == 0:
            return

        if not wysycenie:
            if operation == 'multiply':
                # Normalizacja, max == 255
                # P * V <= 255. V_max dla 255 to 1.
                # max_P = floor(255 / V)
                if abs(value) > 1:
                    max_P = 255 // abs(value)
                else:
                    max_P = 255

                # Wzór: P_new = P_old * (max_P / 255)
                # Następnie: P_new * V, co da nam max 255.
                scale_factor = max_P / 255.0
                pixels = [int(p * scale_factor) for p in pixels]


        for p in pixels:

            if operation == 'add':
                val = p + value
            elif operation == 'multiply':
                val = p * value
            elif operation == 'divide':
                val = p / value
            else:
                continue

            # 2. Zastosowanie wysycenia/normalizacji (obcięcie)
            if wysycenie or operation == 'add':
                val = max(0, min(255, int(round(val))))
            else:
                # Brak wysycenia/skalowanie (dla mnożenia/dzielenia, gdzie normalizacja
                # została wykonana wstępnie lub jest naturalna).
                # Nadal musimy pilnować zakresu 0-255.
                val = max(0, min(255, int(round(val))))

            result_pixels.append(val)

        new_img = Image.new("L", img.size)
        new_img.putdata(result_pixels)

        self.images.append(new_img.convert("RGB"))
        self.isGrayscale.append(True)
        self.current_image_index = len(self.images) - 1
        self.display_image()

    def setup_absolute_difference(self):
        if not self.images:
            messagebox.showwarning("No Image", "First load an image to be the first argument of the operation.")
            return

        img1 = self.images[self.current_image_index]

        filepath = filedialog.askopenfilename(
            title="Select the second image",
            filetypes=[("Images", "*.bmp *.tif *.png *.jpg")]
        )

        if not filepath:
            return

        try:
            img2 = Image.open(filepath)
        except Exception as e:
            messagebox.showerror("Loading Error", f"Failed to load the second image: {e}")
            return

        # Check size
        if img1.size != img2.size:
            messagebox.showerror("Size Error", "Both images must have identical dimensions.")
            return

        img1_L = img1.convert("L")
        img2_L = img2.convert("L")

        self.calculate_absolute_difference(img1_L, img2_L)

    def calculate_absolute_difference(self, img1, img2):
        width, height = img1.size
        pixels1 = list(img1.getdata())
        pixels2 = list(img2.getdata())
        result_pixels = []

        for p1, p2 in zip(pixels1, pixels2):
            val = abs(p1 - p2)
            result_pixels.append(val)

        result_img = Image.new("L", (width, height))
        result_img.putdata(result_pixels)

        self.images.append(result_img.convert("RGB"))
        self.isGrayscale.append(True)
        self.current_image_index = len(self.images) - 1
        self.display_image()

    def setup_canny_detection(self):
        if not self.images:
            messagebox.showwarning("No Image", "No image to process")
            return

        canny_window = tk.Toplevel(self.root)
        canny_window.title("Canny")
        canny_window.geometry("300x200")

        tk.Label(canny_window, text="Threshold 1:").pack(pady=5)
        entry_t1 = tk.Entry(canny_window)
        entry_t1.insert(0, "100")
        entry_t1.pack(pady=5)

        tk.Label(canny_window, text="Threshold 2:").pack(pady=5)
        entry_t2 = tk.Entry(canny_window)
        entry_t2.insert(0, "200")
        entry_t2.pack(pady=5)

        def apply_canny():
            try:
                t1 = int(entry_t1.get())
                t2 = int(entry_t2.get())
                if t1 < 0 or t2 < 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("error", "whole, positive numbers!")
                return

            canny_window.destroy()

            current_img = self.images[self.current_image_index].convert("L")
            img_np = np.array(current_img)

            edges = cv2.Canny(img_np, t1, t2)

            new_img = Image.fromarray(edges)

            self.images.append(new_img.convert("RGB"))
            self.isGrayscale.append(True)
            self.current_image_index = len(self.images) - 1
            self.display_image()

        tk.Button(canny_window, text="Apply", command=apply_canny).pack(pady=20)

    def setup_neighbor_operation(self, operation_type):
        if not self.images:
            messagebox.showwarning("No Image", "No image to process")
            return

        current_img = self.images[self.current_image_index]
        current_img = current_img.convert("L")

        border_value_str = simpledialog.askstring("BORDER_CONSTANT:", "BORDER_CONSTANT:", parent=self.root)

        if not border_value_str:
            return

        try:
            border_value = int(border_value_str)
            if not (0 <= border_value <= 255):
                messagebox.showerror("error", "value must be in range 0-255.")
                return
        except ValueError:
            messagebox.showerror("error", "wrong format")
            return

        border_window = tk.Toplevel(self.root)
        tk.Label(border_window, text="Select border mode:").pack(pady=10)

        border_modes = {
            "Constant": "constant",
            "After": "after",
            "Reflect": "reflect"
        }

        selected_mode = tk.StringVar(border_window, "constant")

        def apply_selection():
            mode = selected_mode.get()
            border_window.destroy()

            if operation_type == 'average':
                self.perform_average_filter(current_img, border_value, mode)
            elif operation_type == 'weighted':
                kernel = self.spawn_kernel_input_window()
                if kernel is not None:
                    self.perform_weighted_filter(current_img, border_value, kernel, mode)
            else:
                self.perform_gaussowski_filter(current_img, border_value, mode)

        for label, value in border_modes.items():
            tk.Radiobutton(border_window, text=label, variable=selected_mode, value=value).pack(anchor="w")

        tk.Button(border_window, text="Apply", command=apply_selection).pack(pady=10)

    def perform_average_filter(self, img, border_value, mode):
        if not self.images:
            return

        try:
            img_np = np.array(img)

            kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.float32)

            kernel_sum = np.sum(kernel)
            if kernel_sum == 0:
                kernel_sum = 1

            kernel = kernel / kernel_sum

            if mode == "constant":
                img_padded_np = cv2.copyMakeBorder(
                    src=img_np,
                    top=1,
                    bottom=1,
                    left=1,
                    right=1,
                    borderType=cv2.BORDER_CONSTANT,
                    value=border_value
                )
                filtered_padded_np = cv2.filter2D(
                    src=img_padded_np,
                    ddepth=-1,
                    kernel=kernel
                )
            elif mode == "after":
                img_padded_np = cv2.filter2D(img_np, -1, kernel)
                img_padded_np[0, :] = 1
                img_padded_np[-1, :] = 1
                img_padded_np[:, 0] = 1
                img_padded_np[:, -1] = 1
                filtered_padded_np = img_padded_np
            else:
                img_padded_np = cv2.copyMakeBorder(
                    src=img_np,
                    top=1,
                    bottom=1,
                    left=1,
                    right=1,
                    borderType=cv2.BORDER_REFLECT,
                    value=border_value
                )
                filtered_padded_np = cv2.filter2D(
                    src=img_padded_np,
                    ddepth=-1,
                    kernel=kernel
                )

            new_img = Image.fromarray(filtered_padded_np.astype(np.uint8))

            self.images.append(new_img.convert("RGB"))
            self.isGrayscale.append(True)
            self.current_image_index = len(self.images) - 1
            self.display_image()

        except Exception as e:
            messagebox.showerror("error", f"{e}")

    def perform_gaussowski_filter(self, img, border_value, mode):
        if not self.images:
            return

        try:
            img_np = np.array(img)

            kernel = np.array([
                [1, 2, 1],
                [2, 5, 2],
                [1, 2, 1]
            ], dtype=np.float32)

            kernel_sum = np.sum(kernel)

            normalized_kernel = kernel / kernel_sum

            padding_size = 1

            if mode == "constant":
                img_padded_np = cv2.copyMakeBorder(
                    src=img_np,
                    top=padding_size,
                    bottom=padding_size,
                    left=padding_size,
                    right=padding_size,
                    borderType=cv2.BORDER_CONSTANT,
                    value=border_value
                )
                filtered_padded_np = cv2.filter2D(
                    src=img_padded_np,
                    ddepth=-1,
                    kernel=normalized_kernel
                )

            elif mode == "after":
                filtered_padded_np = cv2.filter2D(
                    src=img_np,
                    ddepth=-1,
                    kernel=normalized_kernel,
                    borderType=cv2.BORDER_REPLICATE
                )

                filtered_padded_np[0, :] = border_value
                filtered_padded_np[-1, :] = border_value
                filtered_padded_np[:, 0] = border_value
                filtered_padded_np[:, -1] = border_value

            else:
                img_padded_np = cv2.copyMakeBorder(
                    src=img_np,
                    top=padding_size,
                    bottom=padding_size,
                    left=padding_size,
                    right=padding_size,
                    borderType=cv2.BORDER_REFLECT,
                    value=border_value
                )
                filtered_padded_np = cv2.filter2D(
                    src=img_padded_np,
                    ddepth=-1,
                    kernel=normalized_kernel
                )

            new_img = Image.fromarray(filtered_padded_np.astype(np.uint8))

            self.images.append(new_img.convert("RGB"))
            self.isGrayscale.append(True)
            self.current_image_index = len(self.images) - 1
            self.display_image()

        except Exception as e:
            messagebox.showerror("error", f"{e}")

    def perform_weighted_filter(self, img, border_value, kernel, mode):
        if not self.images:
            return

        try:
            img_np = np.array(img)

            kernel_sum = np.sum(kernel)
            if kernel_sum == 0:
                kernel_sum = 1

            normalized_kernel = kernel.astype(np.float32) / kernel_sum

            if mode == "constant":
                img_padded_np = cv2.copyMakeBorder(
                    src=img_np,
                    top=1,
                    bottom=1,
                    left=1,
                    right=1,
                    borderType=cv2.BORDER_CONSTANT,
                    value=border_value
                )
                filtered_padded_np = cv2.filter2D(
                    src=img_padded_np,
                    ddepth=-1,
                    kernel=normalized_kernel
                )

            elif mode == "after":
                filtered_padded_np = cv2.filter2D(
                    src=img_np,
                    ddepth=-1,
                    kernel=normalized_kernel,
                    borderType=cv2.BORDER_REPLICATE
                )

                filtered_padded_np[0, :] = border_value
                filtered_padded_np[-1, :] = border_value
                filtered_padded_np[:, 0] = border_value
                filtered_padded_np[:, -1] = border_value

            else:
                img_padded_np = cv2.copyMakeBorder(
                    src=img_np,
                    top=1,
                    bottom=1,
                    left=1,
                    right=1,
                    borderType=cv2.BORDER_REFLECT,
                    value=border_value
                )
                filtered_padded_np = cv2.filter2D(
                    src=img_padded_np,
                    ddepth=-1,
                    kernel=normalized_kernel
                )

            new_img = Image.fromarray(filtered_padded_np.astype(np.uint8))

            self.images.append(new_img.convert("RGB"))
            self.isGrayscale.append(True)
            self.current_image_index = len(self.images) - 1
            self.display_image()

        except Exception as e:
            messagebox.showerror("Error", f"{e}")

    def spawn_kernel_input_window(self, default_kernel=[1, 1, 1, 1, 1, 1, 1, 1, 1]):
        kernel_window = tk.Toplevel(self.root)
        kernel_window.transient(self.root)
        kernel_window.grab_set()

        entries = []
        kernel_result = None

        for i in range(3):
            for j in range(3):
                entry = tk.Entry(kernel_window, width=5)
                entry.grid(row=i, column=j, padx=5, pady=5)
                entries.append(entry)

        def on_ok():
            nonlocal kernel_result
            try:
                kernel_values = [float(entry.get()) for entry in entries]
                kernel_result = np.array(kernel_values, dtype=np.float32)
                kernel_window.destroy()

            except ValueError:
                messagebox.showerror("Error", "")

        ok_button = tk.Button(kernel_window, text="OK", command=on_ok)
        ok_button.grid(row=3, column=0, columnspan=3, pady=10)

        for entry, value in zip(entries, default_kernel):
            entry.insert(0, str(value))

        self.root.wait_window(kernel_window)

        return kernel_result

    def setup_sobel_detection(self):
        if not self.images:
            messagebox.showwarning("No Image", "No image to process")
            return

        current_img = self.images[self.current_image_index]
        current_img_L = current_img.convert("L")

        border_value_str = simpledialog.askstring("BORDER_CONSTANT:", "BORDER_CONSTANT:", parent=self.root)
        if not border_value_str:
            return

        try:
            border_value = int(border_value_str)
            if not (0 <= border_value <= 255):
                messagebox.showerror("Error", "0-255.")
                return
        except ValueError:
            messagebox.showerror("Error", "Whole number")
            return

        self.sobel_edge_detection(current_img_L, border_value)

    def setup_prewitt_detection(self):
        if not self.images:
            messagebox.showwarning("No Image", "No image to process")
            return

        current_img = self.images[self.current_image_index]
        current_img_L = current_img.convert("L")

        dialog_window = tk.Toplevel(self.root)
        selected_mode = tk.StringVar(dialog_window, "constant")
        border_value_var = tk.StringVar(dialog_window, "0")

        def apply_settings():
            mode = selected_mode.get()
            border_value_str = border_value_var.get()

            try:
                border_value = int(border_value_str)
                if not (0 <= border_value <= 255):
                    messagebox.showerror("Error", "0-255.", parent=dialog_window)
                    return
            except ValueError:
                messagebox.showerror("Error", "whole number.", parent=dialog_window)
                return

            dialog_window.destroy()
            self.prewitt_select_direction(current_img_L, border_value, mode)

        border_frame = tk.Frame(dialog_window, padx=10, pady=10)
        border_frame.pack(fill='x')

        tk.Label(border_frame, text="BORDER_CONSTANT 0-255:").pack(side=tk.LEFT)
        tk.Entry(border_frame, textvariable=border_value_var, width=5).pack(side=tk.LEFT, padx=5)

        mode_frame = tk.LabelFrame(dialog_window, text="Mode", padx=10, pady=10)
        mode_frame.pack(padx=10, pady=5, fill='x')

        modes = {
            "Constant": "constant",
            "Reflect": "reflect",
            "After": "after"
        }

        for label, value in modes.items():
            tk.Radiobutton(
                mode_frame,
                text=label,
                variable=selected_mode,
                value=value
            ).pack(anchor="w")

        tk.Button(dialog_window, text="Set", command=apply_settings).pack(pady=10)

    def prewitt_select_direction(self, img, border_value, mode):
        direction_window = tk.Toplevel(self.root)
        tk.Label(direction_window, text="Direction::").pack(
            pady=10)

        directions = {
            "0° (vertical)": "0",
            "45°": "45",
            "90° (horizontal)": "90",
            "135°": "135",
            "180°": "180",
            "225°": "225",
            "270°": "270",
            "315°": "315"
        }

        selected_direction = tk.StringVar(direction_window, "0")

        def apply_selection():
            angle = selected_direction.get()
            direction_window.destroy()
            self.perform_prewitt_detection(img, border_value, angle, mode)

        for label, value in directions.items():
            tk.Radiobutton(direction_window, text=label, variable=selected_direction, value=value).pack(anchor="w")

        tk.Button(direction_window, text="Set", command=apply_selection).pack(pady=10)

    def perform_prewitt_detection(self, img, border_value, angle_str, mode):
        if not self.images:
            return

        try:
            img_np = np.array(img, dtype=np.float32)

            angle = int(angle_str)
            kernel = None
            if angle == 0:
                kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
            elif angle == 45:
                kernel = np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]], dtype=np.float32)
            elif angle == 90:
                kernel = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
            elif angle == 135:
                kernel = np.array([[1, 1, 0], [1, 0, -1], [0, -1, -1]], dtype=np.float32)
            elif angle == 180:
                kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
            elif angle == 225:
                kernel = np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]], dtype=np.float32)
            elif angle == 270:
                kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
            else:
                kernel = np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]], dtype=np.float32)

            padding_size = 1

            if mode == "constant":
                img_processed_np = cv2.copyMakeBorder(
                    src=img_np,
                    top=padding_size,
                    bottom=padding_size,
                    left=padding_size,
                    right=padding_size,
                    borderType=cv2.BORDER_CONSTANT,
                    value=border_value
                )

                filtered_np = cv2.filter2D(
                    src=img_processed_np,
                    ddepth=-1,
                    kernel=kernel
                )

            elif mode == "after":
                filtered_np = cv2.filter2D(
                    src=img_np,
                    ddepth=-1,
                    kernel=kernel,
                    borderType=cv2.BORDER_REPLICATE
                )

                filtered_np[0, :] = border_value
                filtered_np[-1, :] = border_value
                filtered_np[:, 0] = border_value
                filtered_np[:, -1] = border_value

            else:
                img_processed_np = cv2.copyMakeBorder(
                    src=img_np,
                    top=padding_size,
                    bottom=padding_size,
                    left=padding_size,
                    right=padding_size,
                    borderType=cv2.BORDER_REFLECT,
                    value=border_value
                )

                filtered_np = cv2.filter2D(
                    src=img_processed_np,
                    ddepth=-1,
                    kernel=kernel
                )

            magnitude = np.abs(filtered_np)

            max_val = np.max(magnitude)
            if max_val > 0:
                normalized_magnitude = (magnitude / max_val) * 255
            else:
                normalized_magnitude = magnitude

            result_np = normalized_magnitude.astype(np.uint8)

            new_img = Image.fromarray(result_np)

            self.images.append(new_img.convert("RGB"))
            self.isGrayscale.append(True)
            self.current_image_index = len(self.images) - 1
            self.display_image()

        except Exception as e:
            messagebox.showerror("Error", f"{e}")

    def sobel_edge_detection(self, img, border_value):
        try:
            img_np = np.array(img, dtype=np.float32)

            # 2. Definicja masek Sobela
            # Maska dla gradientu w kierunku X (wykrywa krawędzie pionowe)
            kernel_x = np.array([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ], dtype=np.float32)

            # Maska dla gradientu w kierunku Y (wykrywa krawędzie poziome)
            kernel_y = np.array([
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]
            ], dtype=np.float32)

            img_padded_np = cv2.copyMakeBorder(
                src=img_np,
                top=1,
                bottom=1,
                left=1,
                right=1,
                borderType=cv2.BORDER_CONSTANT,
                value=border_value
            )

            G_x = cv2.filter2D(
                src=img_padded_np,
                ddepth=-1,
                kernel=kernel_x
            )

            G_y = cv2.filter2D(
                src=img_padded_np,
                ddepth=-1,
                kernel=kernel_y
            )

            # 5. Obliczenie Magnitudy Gradientu
            # Magnituda M = sqrt(G_x^2 + G_y^2)
            # Należy użyć funkcji np.hypot, która jest numerycznie stabilniejsza niż ręczne sqrt(a**2 + b**2)
            magnitude = np.hypot(G_x, G_y)

            # 6. Normalizacja i skalowanie do zakresu 0-255 (uint8)
            # Maksymalna możliwa wartość magnitudy Sobela dla obrazu uint8 wynosi ok. 1442.
            # Musimy przeskalować wynik, aby mieścił się w zakresie 0-255.
            # Najprostsza metoda: normalizacja do max/min, a następnie skalowanie

            # Wartości muszą być > 0, aby można było je przeskalować.
            max_val = np.max(magnitude)
            if max_val > 0:
                # Skalowanie: M_skala = (M / max_val) * 255
                normalized_magnitude = (magnitude / max_val) * 255
            else:
                normalized_magnitude = magnitude  # Zostaje 0

            result_np = normalized_magnitude.astype(np.uint8)

            new_img = Image.fromarray(result_np)

            self.images.append(new_img.convert("RGB"))
            self.isGrayscale.append(True)
            self.current_image_index = len(self.images) - 1
            self.display_image()
        except Exception as e:
            messagebox.showerror("Error", f"{e}")

    def setup_linear_sharpening(self):
        if not self.images:
            messagebox.showwarning("No Image", "No image to process")
            return

        current_img = self.images[self.current_image_index].convert("L")

        direction_window = tk.Toplevel(self.root)

        laplasj = {
            "1": [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
            "2": [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
            "3": [[-1, 2, -1], [2, -4, 2], [-1, 2, -1]],
        }
        selected_direction = tk.StringVar(direction_window, "1")

        def apply_selection():
            key = selected_direction.get()
            kernel = np.array(laplasj[key], dtype=np.float32)

            direction_window.destroy()

            try:
                # Convert to float for processing
                img_np = np.array(current_img, dtype=np.float32)

                # Apply Laplacian filter
                lap = cv2.filter2D(
                    src=img_np,
                    ddepth=cv2.CV_32F,
                    kernel=kernel,
                    borderType=cv2.BORDER_REPLICATE
                )

                # Sharpen by subtracting Laplacian (no alpha scaling)
                sharpened = img_np - lap
                sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

                new_img = Image.fromarray(sharpened, mode="L")

                self.images.append(new_img)
                self.isGrayscale.append(True)
                self.current_image_index = len(self.images) - 1
                self.display_image()

            except Exception as e:
                messagebox.showerror("Error", f"{e}")

        for key, value_list in laplasj.items():
            mask_str = "\n".join(" ".join(str(v) for v in row) for row in value_list)
            tk.Radiobutton(
                direction_window,
                text=f"Mask {key}:\n{mask_str}",
                variable=selected_direction,
                value=key
            ).pack(anchor="w")

        tk.Button(direction_window, text="Set", command=apply_selection).pack(pady=10)

    def setup_median_filter(self):
        if not self.images:
            messagebox.showwarning("No Image", "No image to process")
            return

        current_img = self.images[self.current_image_index]
        current_img_L = current_img.convert("L")

        dialog_window = tk.Toplevel(self.root)
        selected_mode = tk.StringVar(dialog_window, "constant")
        border_value_var = tk.StringVar(dialog_window, "0")
        kernel_size_var = tk.IntVar(dialog_window, 3)

        def apply_settings():
            mode = selected_mode.get()
            border_value_str = border_value_var.get()
            kernel_size = kernel_size_var.get()

            try:
                border_value = int(border_value_str)
                if not (0 <= border_value <= 255):
                    messagebox.showerror("Error", "0-255.", parent=dialog_window)
                    return
            except ValueError:
                messagebox.showerror("Error", "whole number", parent=dialog_window)
                return

            dialog_window.destroy()
            self.perform_median_filter(current_img_L, border_value, mode, kernel_size)

        border_frame = tk.Frame(dialog_window, padx=10, pady=10)
        border_frame.pack(fill='x')
        tk.Label(border_frame, text="BORDER_CONSTANT 0-255:").pack(side=tk.LEFT)
        tk.Entry(border_frame, textvariable=border_value_var, width=5).pack(side=tk.LEFT, padx=5)

        mode_frame = tk.LabelFrame(dialog_window, text="Mode", padx=10, pady=10)
        mode_frame.pack(padx=10, pady=5, fill='x')
        modes = {"Constant": "constant", "Reflect": "reflect", "After": "after"}
        for label, value in modes.items():
            tk.Radiobutton(mode_frame, text=label, variable=selected_mode, value=value).pack(anchor="w")

        kernel_frame = tk.LabelFrame(dialog_window, text="Kernel", padx=10, pady=10)
        kernel_frame.pack(padx=10, pady=5, fill='x')

        # suwak :D
        tk.Scale(kernel_frame, from_=3, to=9, resolution=2, orient=tk.HORIZONTAL,
                 variable=kernel_size_var, label="").pack(fill='x')

        tk.Button(dialog_window, text="Set", command=apply_settings).pack(pady=10)

    def perform_median_filter(self, img, border_value, mode, kernel_size):
        if not self.images:
            return

        try:
            img_np = np.array(img, dtype=np.uint8)

            if mode == "constant":
                img_processed_np = cv2.copyMakeBorder(
                    src=img_np,
                    top=kernel_size // 2,
                    bottom=kernel_size // 2,
                    left=kernel_size // 2,
                    right=kernel_size // 2,
                    borderType=cv2.BORDER_CONSTANT,
                    value=border_value
                )
                filtered_np = cv2.medianBlur(img_processed_np, kernel_size)

            elif mode == "after":
                filtered_np = cv2.medianBlur(img_np, kernel_size)
                filtered_np[0, :] = border_value
                filtered_np[-1, :] = border_value
                filtered_np[:, 0] = border_value
                filtered_np[:, -1] = border_value

            else:
                img_processed_np = cv2.copyMakeBorder(
                    src=img_np,
                    top=kernel_size // 2,
                    bottom=kernel_size // 2,
                    left=kernel_size // 2,
                    right=kernel_size // 2,
                    borderType=cv2.BORDER_REFLECT,
                    value=border_value
                )
                filtered_np = cv2.medianBlur(img_processed_np, kernel_size)

            new_img = Image.fromarray(filtered_np)
            self.images.append(new_img.convert("RGB"))
            self.isGrayscale.append(True)
            self.current_image_index = len(self.images) - 1
            self.display_image()

        except Exception as e:
            messagebox.showerror("Error", f"{e}")

    def setup_logic(self, needs_second_image=True):
        if not self.images:
            messagebox.showwarning("No Image", "Load the first image first.")
            return None, None, None, False

        img1 = self.images[self.current_image_index].convert("L")
        img1_np = np.array(img1)

        img2_np = None
        if needs_second_image:
            filepath = filedialog.askopenfilename(
                title="Select second image",
                filetypes=[("Images", "*.bmp *.tif *.png *.jpg")]
            )
            if not filepath:
                return None, None, None, False

            try:
                img2 = Image.open(filepath).convert("L")
                if img1.size != img2.size:
                    messagebox.showerror("Size Error", "Images must have the same dimensions.")
                    return None, None, None, False
                img2_np = np.array(img2)
            except Exception as e:
                messagebox.showerror("Error", f"Could not load second image: {e}")
                return None, None, None, False

        binarize = messagebox.askyesno("Mode", "Convert to Binary?")

        if binarize:
            _, img1_np = cv2.threshold(img1_np, 127, 255, cv2.THRESH_BINARY)
            if img2_np is not None:
                _, img2_np = cv2.threshold(img2_np, 127, 255, cv2.THRESH_BINARY)
        else:
            img1_np = img1_np.astype(np.uint8, copy=False)
            if img2_np is not None:
                img2_np = img2_np.astype(np.uint8, copy=False)

        return img1_np, img2_np, binarize, True

    def perform_not(self):
        img1_np, _, binarize, success = self.setup_logic(needs_second_image=False)
        if not success:
            return

        if binarize:
            result_np = cv2.bitwise_not(img1_np)
        else:
            result_np = cv2.bitwise_not(img1_np)

        new_img = Image.fromarray(result_np, mode="L")
        self.images.append(new_img)
        self.isGrayscale.append(True)
        self.current_image_index = len(self.images) - 1
        self.display_image()

    def perform_and(self):
        img1_np, img2_np, binarize, success = self.setup_logic(needs_second_image=True)
        if not success:
            return

        if binarize:
            result_np = cv2.bitwise_and(img1_np, img2_np)
        else:
            result_np = np.minimum(img1_np, img2_np).astype(np.uint8)

        new_img = Image.fromarray(result_np, mode="L")
        self.images.append(new_img)
        self.isGrayscale.append(True)
        self.current_image_index = len(self.images) - 1
        self.display_image()

    def perform_or(self):
        img1_np, img2_np, binarize, success = self.setup_logic(needs_second_image=True)
        if not success:
            return

        if binarize:
            result_np = cv2.bitwise_or(img1_np, img2_np)
            new_img = Image.fromarray(result_np, mode="L")
            self.images.append(new_img)
            self.isGrayscale.append(True)
        else:
            img1_np = img1_np.astype(np.uint8, copy=False)
            img2_np = img2_np.astype(np.uint8, copy=False)
            result_np = np.maximum(img1_np, img2_np)
            new_img = Image.fromarray(result_np, mode="L")
            self.images.append(new_img)
            self.isGrayscale.append(True)

        self.current_image_index = len(self.images) - 1
        self.display_image()

    def perform_xor(self):
        img1_np, img2_np, binarize, success = self.setup_logic(needs_second_image=True)
        if not success:
            return

        if binarize:
            result_np = cv2.bitwise_xor(img1_np, img2_np)
            new_img = Image.fromarray(result_np, mode="L")
            self.images.append(new_img)
            self.isGrayscale.append(True)
        else:
            img1_np = img1_np.astype(np.uint8, copy=False)
            img2_np = img2_np.astype(np.uint8, copy=False)
            result_np = cv2.absdiff(img1_np, img2_np)
            new_img = Image.fromarray(result_np, mode="L")
            self.images.append(new_img)
            self.isGrayscale.append(True)

        self.current_image_index = len(self.images) - 1
        self.display_image()

    def setup_custom_stretch(self):
        if not self.images:
            messagebox.showwarning("No Image", "No image to process")
            return

        window = tk.Toplevel(self.root)
        window.geometry("300x350")

        tk.Label(window, text="p1:").pack()
        e_p1 = tk.Entry(window)
        e_p1.insert(0, "50")
        e_p1.pack()

        tk.Label(window, text="p2:").pack()
        e_p2 = tk.Entry(window)
        e_p2.insert(0, "200")
        e_p2.pack()

        tk.Label(window, text="q3:").pack()
        e_q3 = tk.Entry(window)
        e_q3.insert(0, "0")
        e_q3.pack()

        tk.Label(window, text="q4:").pack()
        e_q4 = tk.Entry(window)
        e_q4.insert(0, "255")
        e_q4.pack()

        def apply():
            try:
                p1 = int(e_p1.get())
                p2 = int(e_p2.get())
                q3 = int(e_q3.get())
                q4 = int(e_q4.get())

                if p1 >= p2 or q3 >= q4:
                    messagebox.showerror("Error", "p1 < p2 && q3 < q4")
                    return
                if any(v < 0 or v > 255 for v in [p1, p2, q3, q4]):
                    messagebox.showerror("Error", "0-255")
                    return

            except ValueError:
                messagebox.showerror("Error", "Whole numbers")
                return

            window.destroy()

            img = self.images[self.current_image_index].convert("L")
            pixels = list(img.getdata())

            # https://www.algorytm.org/przetwarzanie-obrazow/histogram-rozciaganie/rozciaganie-cs.html
            # jak na wykładzie, lookup table jest szybsze
            lut = []
            scale_factor = (q4 - q3) / (p2 - p1)
            for val in range(256):
                if val < p1:
                    new_val = q3
                elif val > p2:
                    new_val = q4
                else:
                    new_val = (val - p1) * scale_factor + q3

                lut.append(max(0, min(255, int(new_val))))

            new_pixels = [lut[p] for p in pixels]

            new_img = Image.new("L", img.size)
            new_img.putdata(new_pixels)

            self.images.append(new_img.convert("RGB"))
            self.isGrayscale.append(True)
            self.current_image_index = len(self.images) - 1
            self.display_image()

        tk.Button(window, text="Set", command=apply).pack(pady=20)

    def setup_two_value_segment(self):
        if not self.images:
            messagebox.showwarning("No Image", "No image to process")
            return

        seg_window = tk.Toplevel(self.root)
        seg_window.geometry("300x250")

        tk.Label(seg_window, text="T1:").pack()
        e_t1 = tk.Entry(seg_window)
        e_t1.insert(0, "100")
        e_t1.pack(pady=5)

        tk.Label(seg_window, text="T2:").pack()
        e_t2 = tk.Entry(seg_window)
        e_t2.insert(0, "200")
        e_t2.pack(pady=5)

        def apply_segmentation():
            try:
                t1 = int(e_t1.get())
                t2 = int(e_t2.get())

                if t1 < 0 or t2 > 255:
                    messagebox.showerror("Error", "0-255")
                    return
                if t1 >= t2:
                    messagebox.showerror("Error", "T1<T2")
                    return

            except ValueError:
                messagebox.showerror("Error", "Whole numbers")
                return

            seg_window.destroy()

            img = self.images[self.current_image_index].convert("L")
            pixels = list(img.getdata())

            lut = []
            for i in range(256):
                if t1 <= i <= t2:
                    lut.append(255)
                else:
                    lut.append(0)

            new_pixels = [lut[p] for p in pixels]

            new_img = Image.new("L", img.size)
            new_img.putdata(new_pixels)

            self.images.append(new_img.convert("RGB"))
            self.isGrayscale.append(True)
            self.current_image_index = len(self.images) - 1
            self.display_image()

        tk.Button(seg_window, text="Set", command=apply_segmentation).pack(pady=20)

    def setup_otsu_thresholding(self, display_ret=True):
        if not self.images:
            messagebox.showwarning("No Image", "No image to process")
            return

        current_img = self.images[self.current_image_index]

        self.perform_otsu_thresholding(current_img.convert("L"), display_ret)

    def perform_otsu_thresholding(self, img, display_ret):
        try:
            img_np = np.array(img)

            ret, thresh_img = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            new_img = Image.fromarray(thresh_img)

            self.images.append(new_img.convert("RGB"))
            self.isGrayscale.append(True)
            self.current_image_index = len(self.images) - 1
            self.display_image()
            if display_ret:
                messagebox.showinfo("ret value", f"{ret}")

        except Exception as e:
            messagebox.showerror("Error", f"{e}")

    def setup_adaptive_threshold(self):
        if not self.images:
            messagebox.showwarning("No Image", "No image to process")
            return

        window = tk.Toplevel(self.root)
        window.geometry("300x350")
        method_var = tk.StringVar(value="mean")

        tk.Label(window, text="Method:").pack(pady=(10, 5))
        tk.Radiobutton(window, text="Mean", variable=method_var, value="mean").pack(pady=(10, 5))
        tk.Radiobutton(window, text="Gaussian", variable=method_var, value="gaussian").pack(pady=(10, 5))

        tk.Frame(window, height=2, bd=1, relief="sunken").pack(fill="x", padx=5, pady=10)

        tk.Label(window, text="Block Size:").pack()
        e_block = tk.Entry(window)
        e_block.insert(0, "11")
        e_block.pack()

        tk.Label(window, text="Constant C:").pack()
        e_c = tk.Entry(window)
        e_c.insert(0, "2")
        e_c.pack()

        def apply_adaptive():
            try:
                block_size = int(e_block.get())
                c_val = int(e_c.get())
                method_str = method_var.get()

                if block_size <= 1 or block_size % 2 == 0:
                    messagebox.showerror("Error", "Block Size must be uneven & > 1")
                    return

            except ValueError:
                messagebox.showerror("Error", "whole numbers")
                return

            window.destroy()

            try:
                current_img = self.images[self.current_image_index]
                current_img_L = current_img.convert("L")
                img_np = np.array(current_img_L)

                adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C
                if method_str == "gaussian":
                    adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C

                result_np = cv2.adaptiveThreshold(
                    img_np,
                    255,
                    adaptive_method,
                    cv2.THRESH_BINARY,
                    block_size,
                    c_val
                )

                new_img = Image.fromarray(result_np)
                self.images.append(new_img.convert("RGB"))
                self.isGrayscale.append(True)
                self.current_image_index = len(self.images) - 1
                self.display_image()

            except Exception as e:
                messagebox.showerror("Error", f"{e}")

        tk.Button(window, text="Apply", command=apply_adaptive, height=2, width=15).pack(pady=20)

    def setup_morphology_operation(self):
        if not self.images:
            messagebox.showwarning("No Image", "No image to process")
            return

        morph_window = tk.Toplevel(self.root)
        morph_window.geometry("300x350")

        operation_var = tk.StringVar(value="erosion")
        shape_var = tk.StringVar(value="rect")

        lbl_op = tk.Label(morph_window, text="operation:")
        lbl_op.pack(pady=(10, 5))

        ops = [
            ("erosion", "erosion"),
            ("dilation", "dilation"),
            ("opening", "opening"),
            ("closing", "closing")
        ]

        for text, value in ops:
            tk.Radiobutton(morph_window, text=text, variable=operation_var, value=value).pack(anchor="w", padx=20)

        tk.Frame(morph_window, height=2, bd=1, relief="sunken").pack(fill="x", padx=5, pady=10)

        shapes = [
            ("rectangle", "rect"),
            ("cross", "cross")
        ]

        for text, value in shapes:
            tk.Radiobutton(morph_window, text=text, variable=shape_var, value=value).pack(anchor="w", padx=20)

        def apply_morphology():
            op = operation_var.get()
            shape = shape_var.get()
            morph_window.destroy()

            current_img = self.images[self.current_image_index]
            current_img_L = current_img.convert("L")

            self.perform_morphology(current_img_L, op, shape)

        tk.Button(morph_window, text="Set", command=apply_morphology, height=2, width=15).pack(pady=20)

    def perform_morphology(self, img, operation, shape_type):
        try:
            img_np = np.array(img)
            kernel = None

            if shape_type == 'rect':
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            else:
                kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

            result_np = None

            if operation == 'erosion':
                result_np = cv2.erode(img_np, kernel, iterations=1)

            elif operation == 'dilation':
                result_np = cv2.dilate(img_np, kernel, iterations=1)

            elif operation == 'opening':
                result_np = cv2.morphologyEx(img_np, cv2.MORPH_OPEN, kernel)

            else:
                result_np = cv2.morphologyEx(img_np, cv2.MORPH_CLOSE, kernel)

            new_img = Image.fromarray(result_np)
            self.images.append(new_img.convert("RGB"))
            self.isGrayscale.append(True)
            self.current_image_index = len(self.images) - 1
            self.display_image()

        except Exception as e:
            messagebox.showerror("Errir", f"{e}")

    def setup_skeletonization(self):
        if not self.images:
            messagebox.showwarning("Error", "No image")
            return

        skel_window = tk.Toplevel(self.root)
        skel_window.geometry("300x250")

        shape_var = tk.StringVar(value="rect")

        lbl_shape = tk.Label(skel_window, text="Structuring element shape:")
        lbl_shape.pack(pady=(10, 5))

        shapes = [
            ("rectangle", "rect"),
            ("cross", "cross")
        ]

        for text, value in shapes:
            tk.Radiobutton(skel_window, text=text, variable=shape_var, value=value).pack(anchor="w", padx=20)

        def apply_skeletonization():
            shape = shape_var.get()
            skel_window.destroy()

            current_img = self.images[self.current_image_index]
            current_img_L = current_img.convert("L")
            self.perform_skeletonization(current_img_L, shape)

        tk.Button(skel_window, text="Set", command=apply_skeletonization, height=2, width=15).pack(pady=20)

    def perform_skeletonization(self, img, shape_type="cross"):
        try:
            img_np = np.array(img)
            _, binary_img = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            size = np.size(binary_img)
            skel = np.zeros(binary_img.shape, np.uint8)

            if shape_type == 'rect':
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            else:
                kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

            img = binary_img.copy()
            done = False

            while not done:
                eroded = cv2.erode(img, kernel)
                temp = cv2.dilate(eroded, kernel)
                temp = cv2.subtract(img, temp)
                skel = cv2.bitwise_or(skel, temp)
                img = eroded.copy()

                zeros = size - cv2.countNonZero(img)
                if zeros == size:
                    done = True

            new_img = Image.fromarray(skel)
            self.images.append(new_img.convert("RGB"))
            self.isGrayscale.append(True)
            self.current_image_index = len(self.images) - 1
            self.display_image()

        except Exception as e:
            messagebox.showerror("Error", f"{e}")

    def calculate_moments(self):
        if not self.images:
            messagebox.showwarning("No Image", "No image to process")
            return

        # hide the checkbox
        self.setup_otsu_thresholding(display_ret=False)

        img = self.images[self.current_image_index].convert("L")
        arr = np.array(img, dtype=np.uint8)

        _, binary = cv2.threshold(arr, 1, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            messagebox.showwarning("No Contours", "No contours found in image")
            return

        filepath = filedialog.asksaveasfilename(
            title="Save contour moments",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")]
        )
        if not filepath:
            return

        workbook = xlsxwriter.Workbook(filepath)

        try:
            sheet = workbook.add_worksheet("Moments")

            moment_keys = list(cv2.moments(contours[0]).keys())

            sheet.write(0, 0, "ID")
            for col, key in enumerate(moment_keys, start=1):
                sheet.write(0, col, key)
            base_col = len(moment_keys) + 1
            sheet.write(0, base_col, "Area")
            sheet.write(0, base_col + 1, "Perimeter")
            sheet.write(0, base_col + 2, "AspectRatio")
            sheet.write(0, base_col + 3, "Extent")
            sheet.write(0, base_col + 4, "Solidity")
            sheet.write(0, base_col + 5, "EquivalentDiameter")

            # Rows
            for idx, contour in enumerate(contours, start=1):
                moments = cv2.moments(contour)
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)

                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h != 0 else 0

                rect_area = w * h
                extent = area / rect_area if rect_area != 0 else 0

                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area != 0 else 0

                equivalent_diameter = np.sqrt(4 * area / np.pi) if area > 0 else 0

                sheet.write(idx, 0, f"id_{idx}")
                for col, key in enumerate(moment_keys, start=1):
                    sheet.write(idx, col, f"{moments[key]:.25f}")
                sheet.write(idx, base_col, f"{area:.2f}")
                sheet.write(idx, base_col + 1, f"{perimeter:.2f}")
                sheet.write(idx, base_col + 2, f"{aspect_ratio:.4f}")
                sheet.write(idx, base_col + 3, f"{extent:.4f}")
                sheet.write(idx, base_col + 4, f"{solidity:.4f}")
                sheet.write(idx, base_col + 5, f"{equivalent_diameter:.4f}")

        finally:
            workbook.close()

        messagebox.showinfo("Success", f"Shapes saved to {filepath}")

    def hough_edge_detection(self, canny_threshold1=50, canny_threshold2=150, hough_threshold=100):
        if not self.images:
            return

        try:
            img = self.images[self.current_image_index]
            img_gray = np.array(img.convert("L"), dtype=np.uint8)

            edges = cv2.Canny(img_gray, canny_threshold1, canny_threshold2, apertureSize=3)

            lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180,
                                    threshold=hough_threshold,
                                    minLineLength=30, maxLineGap=10)

            img_color = np.array(img.convert("RGB"))
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(img_color, (x1, y1), (x2, y2), (0, 0, 255), 2)

            new_img = Image.fromarray(img_color)
            self.images.append(new_img)
            self.isGrayscale.append(False)
            self.current_image_index = len(self.images) - 1
            self.display_image()

        except Exception as e:
            messagebox.showerror("Error", f"{e}")


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    first_window = tk.Toplevel()
    app = ImageViewer(first_window)
    root.mainloop()
