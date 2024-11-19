import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import math
import gradio as gr

import matplotlib.pyplot as plt
# %config InlineBackend.figure_format = 'retina'

import ipywidgets as widgets
from IPython.display import display, clear_output

from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
from transformers import BlipProcessor, BlipForConditionalGeneration
from groq import Groq
import re
import json

GROQ_API_KEY = "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"

client = Groq(api_key = GROQ_API_KEY)


processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")


torch.set_grad_enabled(False);

# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# # colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    classes_predicted = []
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        (CLASSES[cl])
        classes_predicted.append(CLASSES[cl])
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()
    return list(set(classes_predicted))

model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
model.eval();


def get_caption(img_url):
  raw_image = Image.open(img_url).convert('RGB')
  inputs = processor(raw_image, return_tensors="pt")

  out = caption_model.generate(**inputs)
  print(processor.decode(out[0], skip_special_tokens=True))
  return str(processor.decode(out[0], skip_special_tokens=True))


def get_objects(url):
  # url = '/content/saved_image.png'
  im = Image.open(url)

  # mean-std normalize the input image (batch-size: 1)
  img = transform(im).unsqueeze(0)

  # propagate through the model
  outputs = model(img)

  # keep only predictions with 0.7+ confidence
  probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
  keep = probas.max(-1).values > 0.9

  # convert boxes from [0; 1] to image scales
  bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

  res = plot_results(im, probas[keep], bboxes_scaled)
  # print(res)
  return res


system_prompt = """
    <SystemPrompt>
            Extract Tags from the provided text.
            The Tags that will be used to search.

        <OutputFormat>
            Format the output in the following JSON structure

            {
                "tags" : [* list of tags here*]
            }

        </OutputFormat>

    </SystemPrompt>

"""
def get_tags(text, objects):
  try:

    user_prompt = f"""
    Extract the Tags from thei text:

    {text}

    {objects}

    """

    chat_completion = client.chat.completions.create(

        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
        model="llama3-8b-8192",
        response_format={"type": "json_object"},

        stream=False,
    )

    print(chat_completion)

    json_data = json.loads(chat_completion.choices[0].message.content)
    return json_data['tags'], chat_completion.usage.total_tokens * 0.00000005

  except Exception as e:
    print(f"Exception | get_tags | {str(e)}")


# Image processing function
def image_to_tags(image):
    # tags = "shahzain, haider"
    image = Image.fromarray(image)
    image.save("saved_image.png")

    generated_caption = get_caption('saved_image.png')
    print(generated_caption)

    objects = get_objects('saved_image.png')

    tags, cost = get_tags(generated_caption, ", ".join(objects))

    return ", ".join(tags) , generated_caption , ", ".join(objects), cost
    # return "", "", ""

# Define Gradio interface
app = gr.Interface(
    fn=image_to_tags,
    inputs=gr.Image(type="numpy", label="Upload an Image"),
 outputs=[
        gr.Label(num_top_classes=5, label="Predicted Tags"),
        gr.Textbox(label="Caption"),
        gr.Textbox(label="Object Detection"),
             gr.Textbox(label="Cost")

    ],    title="Image Tagging App"
)

# Launch the app
app.launch(debug = True, share=True)
