import cv2
import numpy as np
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from ultralytics import YOLO  
import torch
from django.shortcuts import render
from .models import DetectedObject, Score
import time
import json
from django.http import JsonResponse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
import json
from .models import DetectedObject, Score
from deep_translator import GoogleTranslator
import logging
from indic_transliteration import sanscript  # For Indian language transliteration
from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate

logger = logging.getLogger(__name__)

LANG_CODE_MAP = {
    'tamil': 'ta',
    'telugu': 'te',
    'kannada': 'kn',
    'hindi': 'hi',
    'french': 'fr',
    'japanese': 'ja'
}


def index(request):
    return render(request,'index.html')

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.4


detected_words = set()


def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(img)

            detected_names = results.pandas().xyxy[0]['name'].tolist()

            for name in detected_names:
                # Create or update object in DB
                obj, created = DetectedObject.objects.get_or_create(name=name)
                if not obj.seen:
                    obj.seen = True
                    obj.save()

            results.render()
            annotated_frame = results.ims[0]
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def video_feed(request):
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def get_detected_objects(request):
    objects = DetectedObject.objects.filter(seen=True).values_list('name', flat=True)
    return JsonResponse({'objects': list(objects)})

def get_detected_words(request):
    return JsonResponse({'words': list(detected_words)})


def detection(request):
    return render(request, 'det.html')

def list(request):
    objects = DetectedObject.objects.filter(seen=True)
    return render(request, 'list.html', {"detected_objects": objects})

@csrf_exempt
def update_score_and_learn(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            word = data.get('word')
            lang = data.get('lang')

            if not word or not lang:
                return JsonResponse({'status': 'failed', 'error': 'Missing word or language'}, status=400)

            obj = DetectedObject.objects.filter(name__iexact=word).first()
            if obj:
                obj.learnt = True
                if lang not in obj.language_learntin:
                    obj.language_learntin.append(lang)
                obj.save()

                score, _ = Score.objects.get_or_create(id=1)
                score.points += 10
                score.save()

                return JsonResponse({'status': 'success'})
            return JsonResponse({'status': 'failed', 'error': 'Word not found'}, status=404)
        except Exception as e:
            logger.error(f"Error in update_score_and_learn: {str(e)}")
            return JsonResponse({'status': 'failed', 'error': str(e)}, status=500)
    return JsonResponse({'status': 'failed', 'error': 'Invalid method'}, status=405)

def get_pronunciation(text, lang_code):
    """Convert non-English text to English pronunciation"""
    try:
        if lang_code in ['ta', 'te', 'kn', 'hi']:  # Indian languages
            return transliterate(text, sanscript.SCHEMES[lang_code], sanscript.ITRANS)
        elif lang_code == 'ja':  # Japanese
            # Using romaji for Japanese pronunciation
            import pykakasi
            kks = pykakasi.kakasi()
            result = kks.convert(text)
            return ' '.join([item['hepburn'] for item in result])
        else:
            return text  # Return as-is for European languages
    except Exception as e:
        logger.error(f"Transliteration error: {str(e)}")
        return ""

def translate_word(request):
    word = request.GET.get('word', '').strip()
    lang = request.GET.get('lang', '').lower().strip()

    if not word or not lang:
        return JsonResponse({'translated': 'Invalid input', 'pronunciation': ''}, status=400)

    lang_code = LANG_CODE_MAP.get(lang)
    if not lang_code:
        return JsonResponse({'translated': 'Unsupported language', 'pronunciation': ''}, status=400)

    try:
        # Get translation
        translation = GoogleTranslator(source='auto', target=lang_code).translate(word)
        
        # Get pronunciation
        pronunciation = get_pronunciation(translation, lang_code)
        
        return JsonResponse({
            'translated': translation,
            'pronunciation': pronunciation
        })
    except Exception as e:
        logger.error(f"Translation error for word '{word}' to {lang_code}: {str(e)}")
        # Fallback to simple word mapping if translation fails
        simple_data = get_simple_translations(word, lang_code)
        if simple_data:
            return JsonResponse(simple_data)
        return JsonResponse({
            'translated': 'Translation service unavailable',
            'pronunciation': ''
        }, status=503)

def get_simple_translations(word, lang_code):
    """Fallback simple translations for common objects with pronunciation"""
    word = word.lower()
    translations = {
        'apple': {
            'ta': {'translated': 'ஆப்பிள்', 'pronunciation': 'aappil'},
            'te': {'translated': 'ఆపిల్', 'pronunciation': 'aapil'},
            'kn': {'translated': 'ಸೇಬು', 'pronunciation': 'sebu'},
            'hi': {'translated': 'सेब', 'pronunciation': 'seb'},
            'fr': {'translated': 'pomme', 'pronunciation': 'pom'},
            'ja': {'translated': 'りんご', 'pronunciation': 'ringo'}
        },
        'book': {
            'ta': {'translated': 'புத்தகம்', 'pronunciation': 'puththagam'},
            'te': {'translated': 'పుస్తకం', 'pronunciation': 'pustakam'},
            'kn': {'translated': 'ಪುಸ್ತಕ', 'pronunciation': 'pustaka'},
            'hi': {'translated': 'किताब', 'pronunciation': 'kitaab'},
            'fr': {'translated': 'livre', 'pronunciation': 'leevr'},
            'ja': {'translated': '本', 'pronunciation': 'hon'}
        }
        # Add more common words as needed
    }
    return translations.get(word, {}).get(lang_code, {'translated': '', 'pronunciation': ''})

# # Load pre-trained YOLOv5s model from PyTorch Hub
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# model.conf = 0.5  # Confidence threshold for object detection

# IGNORE_CLASS = 0  # Class ID for "person" (you can change this based on your use case)

# def gen_frames():
#     url = '/dev/video10'  # Virtual webcam device created by v4l2loopback
#     cap = cv2.VideoCapture(url)  # Open the virtual webcam
    
#     while True:
#         success, frame = cap.read()  # Capture frame-by-frame
#         if not success:
#             break
#         else:
#             # Convert frame to RGB (required by YOLO)
#             img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
#             # Run YOLO inference
#             results = model(img)
            
#             # Filter out "person" class (ID 0)
#             filtered_results = []
#             for i, label in enumerate(results.xywh[0][:, -1].cpu().numpy()):
#                 if label != IGNORE_CLASS:
#                     filtered_results.append(i)
            
#             # Render (draw) bounding boxes and labels for all but "person"
#             results.render()  # This modifies the image in place
#             for i in filtered_results:
#                 x1, y1, x2, y2 = results.xywh[0][i][:4].cpu().numpy()
#                 label = int(results.xywh[0][i][-1].cpu().numpy())
#                 color = (0, 255, 0)  # Green for non-person classes
#                 cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
#                 cv2.putText(frame, f'{results.names[label]} {results.xywh[0][i][-2].cpu().numpy():.2f}', 
#                             (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#             # Convert back to BGR for OpenCV
#             annotated_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

#             # Encode the frame as JPEG for streaming
#             ret, buffer = cv2.imencode('.jpg', annotated_frame)
#             frame = buffer.tobytes()  # Convert to byte format for streaming

#             # Yield the frame as part of multipart response
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

