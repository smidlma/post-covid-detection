from ultralytics import YOLO

model = YOLO("best.pt")
dataset_dir = "dataset/train/images/"
# images = [
#     "dataset/train/images/410670_3990896_1.3.51.0.7.14131502767.42928.4171.41366.7041.63017.28515.dcm.png",
#     "dataset/train/images/256443_3924866_1.3.51.0.7.12135885350.20836.54593.44985.58450.18728.43052.dcm.png",
#     "08026.jpg",
#     "08051.jpg",
# ]

# images = ["device01.jpeg", "device02.jpg", "device03.jpeg"]

images = [
    "cropped/142704_4012344_1.2.840.113564.10.1.28394376252560817261153186159151834210662.dcm_cropped.png",
    "cropped/256443_3924866_1.3.51.0.7.12135885350.20836.54593.44985.58450.18728.43052.dcm_cropped.png",
]

results = model(source=images, conf=0.4, device="mps", classes=[0, 1])


for i, r in enumerate(results):
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    # im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Show results to screen (in supported environments)
    r.show()

    # Save results to disk
    # r.save(filename=f'results{i}.jpg')
