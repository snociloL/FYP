# -*- coding: utf-8 -*-
"""Builds CSP650 Chapter 5 (Conclusion and Recommendation) as a .docx
matching the Chapter 4 formatting (A4, Times New Roman 12pt, 1.5 spacing)."""
import zipfile, xml.etree.ElementTree as ET, os

OUT = r"C:\Users\sno PC\Documents\UiTM Degree\CSP650_Chapter5_Conclusion_and_Recommendation.docx"


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def run(t, b=False):
    rpr = "<w:rPr><w:b/></w:rPr>" if b else ""
    return f'<w:r>{rpr}<w:t xml:space="preserve">{esc(t)}</w:t></w:r>'


def title(t):
    return ('<w:p><w:pPr><w:jc w:val="center"/>'
            '<w:spacing w:after="240" w:line="360" w:lineRule="auto"/></w:pPr>'
            + run(t, True) + "</w:p>")


def head(t):
    return ('<w:p><w:pPr><w:spacing w:before="240" w:after="120" '
            'w:line="360" w:lineRule="auto"/></w:pPr>' + run(t, True) + "</w:p>")


def body(t):
    return ('<w:p><w:pPr><w:jc w:val="both"/><w:ind w:firstLine="720"/></w:pPr>'
            + run(t) + "</w:p>")


def item(label, t):
    return ('<w:p><w:pPr><w:jc w:val="both"/><w:ind w:left="1080" w:hanging="360"/></w:pPr>'
            + run(label + ".") + "<w:r><w:tab/></w:r>" + run(t) + "</w:p>")


P = []
P.append(title("CHAPTER 5"))
P.append(title("CONCLUSION AND RECOMMENDATION"))
P.append(body("This chapter presents an overall summary of the research project and reflects on the accomplishment of the objectives established in Chapter 1. It revisits the problem the project set out to address, summarises how each objective was achieved through the work reported in the preceding chapters, discusses the limitations encountered during development, and finally offers a set of recommendations and directions for future work that could extend and strengthen the Deepfake Video Validator."))

P.append(head("5.1\tOverall"))
P.append(body("The Deepfake Video Validator, a browser-based system for detecting AI-manipulated (deepfake) videos using a ResNet-50 and Long Short-Term Memory (LSTM) spatiotemporal model, has been successfully developed in order to achieve its objectives. The system enables a user to obtain a deepfake verdict either by uploading a video file or by analysing the video currently playing in the active browser tab, and it presents the result through a simple, colour-coded interface designed for non-technical users."))
P.append(body("The project was carried out through several phases. The first phase, problem identification and preliminary study, was conducted through a review of related literature and existing deepfake-detection approaches in order to build the understanding needed to design the system. The data preparation phase followed, in which raw videos from publicly available datasets, namely the DeepFake Detection Challenge and FaceForensics++, were converted into fixed-length sequences of face crops through a leakage-safe pre-processing pipeline. The model design phase realised the spatiotemporal architecture, pairing a ResNet-50 convolutional backbone, wrapped in a TimeDistributed layer, with a 128-unit LSTM temporal head. The training phase applied a two-stage transfer-learning strategy together with balanced sampling and data augmentation. Finally, the implementation and evaluation phase deployed the trained model behind a Python (Flask) backend hosted on Hugging Face Spaces and connected it to a Manifest V3 browser extension, after which the system was evaluated for classification accuracy, score separation and inference latency, as reported in Chapter 4."))

P.append(head("5.2\tObjective Achievement"))
P.append(body("Three objectives were established for this project in Chapter 1. This section summarises how each of these objectives was accomplished through the work presented in the preceding chapters."))

P.append(head("5.2.1\tFirst Objective"))
P.append(body("The first objective is to develop a robust machine-learning model capable of detecting deepfake video content. This objective was achieved in the data preparation, model design and training phases. A spatiotemporal model was constructed in which a ResNet-50 backbone extracts a 2,048-dimensional feature vector from each of the thirty frames in a sequence, and a 128-unit LSTM layer aggregates these features over time so as to exploit temporal cues such as unnatural motion and flicker that single-frame detectors cannot capture. The model was trained using a two-stage transfer-learning procedure with balanced sampling and augmentation to address class imbalance. As reported in Section 4.6, the trained model achieved a validation accuracy of 88.9% and an F1-score of 92.6%, and it placed authentic and synthetic sequences on opposite sides of the calibrated decision boundary with a mean score separation of 0.477. These results demonstrate that the model is able to discriminate between authentic and manipulated video, thereby achieving the first objective."))

P.append(head("5.2.2\tSecond Objective"))
P.append(body("The second objective is to deploy the model in a low-friction web extension for popular video-sharing platforms. This objective was achieved in the implementation phase. The trained model was served from a Python (Flask) backend hosted on a cloud platform (Hugging Face Spaces), and a Manifest V3 browser extension compatible with Chromium-based browsers such as Google Chrome and Microsoft Edge was developed to consume it. The extension allows a user to obtain a verdict either by uploading a video file or by capturing and analysing the video in the current browser tab, and it presents the outcome through a simple tri-state, colour-coded interface, namely authentic, deepfake or uncertain, accompanied by a confidence value. A passive, auto-hiding on-page status indicator further reduces the effort required of the user. As demonstrated in Section 4.5, the extension provides a working, low-friction route to a deepfake verdict, thereby achieving the second objective."))

P.append(head("5.2.3\tThird Objective"))
P.append(body("The third objective is to rigorously determine the detection accuracy and latency of the system. This objective was achieved in the evaluation phase. The system was assessed using standard classification metrics, an analysis of the separation between the authentic and synthetic score distributions, and an inference-latency benchmark, all of which are reported in Section 4.6. The evaluation produced an accuracy of 88.9%, a precision of 100%, a recall of 86.3% and an F1-score of 92.6%, together with a mean score separation of 0.477. The latency benchmark recorded a per-frame inference time of 81.8 ms and an end-to-end response time of approximately 6.6 seconds per request on the cloud deployment. Taken together, these measurements provide a rounded, quantitative assessment of the system's effectiveness and efficiency, thereby achieving the third objective."))

P.append(head("5.3\tLimitation"))
P.append(body("As the project progressed, several limitations were encountered. First, the quantity and balance of the authentic (real) training and validation videos were limited relative to the manipulated videos; although balanced sampling and augmentation mitigated this imbalance during training, the validation set in particular contained only a small number of authentic sequences, so the reported real-side performance may not fully generalise, and a larger and more balanced dataset would be expected to yield a more robust and representative estimate. Second, because the deployed system performs inference on a remote backend, the latency experienced by the user depends on network conditions and on whether the backend is warm; the per-frame inference time therefore exceeded the real-time target, and the first request made after a period of inactivity incurred a cold-start delay. Third, the face-detection stage relies on a frontal-face Haar cascade and analyses only the single largest face in each frame, so videos containing profile views, heavily occluded faces, or several subjects of interest may not be assessed correctly, and videos in which no face can be detected cannot be analysed at all. Finally, the model was trained on a limited set of manipulation methods, so its ability to generalise to deepfakes produced by unseen or newer generation techniques is not guaranteed."))

P.append(head("5.4\tRecommendation"))
P.append(body("Although the system has been successfully developed and demonstrates the feasibility of browser-based deepfake detection, several improvements are recommended for future researchers and developers:"))
P.append(item("i", "Expand and further balance the training dataset by incorporating a larger number of authentic videos and a wider variety of manipulation methods, which is expected to improve generalisation and reduce false classifications."))
P.append(item("ii", "Host the backend on an always-warm or GPU-backed server, or optimise the model through techniques such as quantisation or knowledge distillation, in order to lower inference latency and remove the cold-start delay so that true real-time analysis becomes feasible."))
P.append(item("iii", "Replace the frontal-face Haar cascade with a modern deep-learning face detector, such as MTCNN or RetinaFace, to handle profile views, occlusions and multiple faces more reliably."))
P.append(item("iv", "Extend browser support beyond Chromium-based browsers, for example by providing a Firefox-compatible build, so that the tool reaches a wider audience."))
P.append(item("v", "Investigate newer model architectures, such as attention-based or transformer-based spatiotemporal networks, to further improve detection accuracy."))
P.append(item("vi", "Add an explainability component, such as visual highlighting of the manipulated facial regions, to increase user trust and provide insight into the basis of each decision."))
P.append(body("In conclusion, the Deepfake Video Validator has met all three of its objectives, delivering a working spatiotemporal detection model and a low-friction browser extension supported by a quantitative evaluation. The limitations identified above, together with the recommendations for future work, provide a clear direction for extending the system into a more accurate, faster and more widely deployable tool for combating the growing threat of deepfake video content."))

sect = ('<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1701" w:right="1417" w:bottom="1701" w:left="2268" '
        'w:header="720" w:footer="720" w:gutter="0"/></w:sectPr>')
document = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            '<w:body>' + "".join(P) + sect + "</w:body></w:document>")

styles = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
          '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
          '<w:docDefaults><w:rPrDefault><w:rPr>'
          '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" '
          'w:eastAsia="Times New Roman" w:cs="Times New Roman"/>'
          '<w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr></w:rPrDefault>'
          '<w:pPrDefault><w:pPr><w:spacing w:after="240" w:line="360" w:lineRule="auto"/></w:pPr></w:pPrDefault>'
          '</w:docDefaults>'
          '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>'
          '</w:styles>')

ct = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
      '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
      '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
      '<Default Extension="xml" ContentType="application/xml"/>'
      '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
      '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
      '</Types>')
rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        '</Relationships>')
drels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
         '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
         '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
         '</Relationships>')

ET.fromstring(document)
ET.fromstring(styles)

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("[Content_Types].xml", ct)
    z.writestr("_rels/.rels", rels)
    z.writestr("word/document.xml", document)
    z.writestr("word/styles.xml", styles)
    z.writestr("word/_rels/document.xml.rels", drels)

print("[OK] wrote", OUT)
print("paragraphs:", len(P), "| size KB:", os.path.getsize(OUT) // 1024)
