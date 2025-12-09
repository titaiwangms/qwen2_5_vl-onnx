# cpu
python builder_od.py -m Qwen/Qwen2.5-VL-7B-Instruct --device cpu --dtype float32 --exporter onnx-dynamo --pretrained --second-input --zip
python builder_od.py -m microsoft/Fara-7B --device cpu --dtype float32 --exporter onnx-dynamo --pretrained --second-input --zip

# cuda
# python builder_od.py -m microsoft/Fara-7B --device cuda --dtype float16 --exporter custom --pretrained --second-input --zip
QWEN25ATTENTION=PACKED python builder_od.py -m Qwen/Qwen2.5-VL-7B-Instruct --device cuda --dtype float16 --exporter onnx-dynamo --pretrained --second-input --zip
QWEN25ATTENTION=PACKED python builder_od.py -m microsoft/Fara-7B --device cuda --dtype float16 --exporter onnx-dynamo --pretrained --second-input --zip
QWEN25ATTENTION=LOOPMHA python builder_od.py -m microsoft/Fara-7B --device cuda --dtype float16 --exporter onnx-dynamo --pretrained --second-input --zip

