#!/usr/bin/env python3
"""
Inference script for fine-tuned Qwen3-VL on Cuprite hyperspectral data.

Usage:
    python inference_cuprite.py --image path/to/image.png --checkpoint path/to/checkpoint
    python inference_cuprite.py --test  # Run on test set
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_model(
    base_model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
    checkpoint_path: Optional[str] = None,
    device: str = "cuda"
):
    """Load base model and optionally merge LoRA weights."""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    
    print(f"Loading base model: {base_model_name}")
    processor = AutoProcessor.from_pretrained(base_model_name)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    
    if checkpoint_path:
        print(f"Loading LoRA weights from: {checkpoint_path}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, checkpoint_path)
        model = model.merge_and_unload()
    
    model.eval()
    return model, processor


def inference_single(
    model,
    processor,
    image_path: str,
    prompt: str = "Does this hyperspectral image show copper-related alteration features? If yes, identify the alteration minerals present.",
    max_new_tokens: int = 256
) -> str:
    """Run inference on a single image."""
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Build messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    # Process inputs
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    ).to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    
    # Decode
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    response = processor.decode(generated_ids, skip_special_tokens=True)
    
    return response


def run_test_set(
    model,
    processor,
    test_json_path: str,
    data_dir: str
):
    """Run inference on the test set and compare with ground truth."""
    
    with open(test_json_path, 'r') as f:
        test_data = json.load(f)
    
    print(f"\nRunning inference on {len(test_data)} test samples...")
    print("=" * 60)
    
    results = []
    for i, item in enumerate(test_data):
        image_path = Path(data_dir) / item['image']
        
        # Extract question from conversations
        question = ""
        expected = ""
        for msg in item['conversations']:
            if msg['from'] == 'human':
                question = msg['value'].replace('<image>\n', '').strip()
            elif msg['from'] == 'gpt':
                expected = msg['value']
        
        # Run inference
        response = inference_single(model, processor, str(image_path), question)
        
        results.append({
            'image': item['image'],
            'question': question,
            'expected': expected,
            'predicted': response
        })
        
        print(f"\n[{i+1}/{len(test_data)}] {item['image']}")
        print(f"Q: {question[:80]}...")
        print(f"Expected: {expected[:80]}...")
        print(f"Predicted: {response[:80]}...")
    
    print("\n" + "=" * 60)
    print("Test complete!")
    
    # Save results
    output_path = Path(data_dir) / "inference_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Cuprite mineral inference")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--prompt", type=str, 
                        default="Does this hyperspectral image show copper-related alteration features?",
                        help="Question to ask")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to LoRA checkpoint")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
                        help="Base model name or path")
    parser.add_argument("--test", action="store_true",
                        help="Run on test set")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if args.checkpoint is None:
        default_checkpoint = PROJECT_ROOT / "output" / "cuprite_finetune"
        if default_checkpoint.exists():
            args.checkpoint = str(default_checkpoint)
            print(f"Using default checkpoint: {args.checkpoint}")
    
    # Load model
    model, processor = load_model(
        args.base_model,
        args.checkpoint,
        args.device
    )
    
    if args.test:
        # Run on test set
        test_json = PROJECT_ROOT / "data" / "cuprite_dataset" / "test_qwenvl.json"
        data_dir = PROJECT_ROOT / "data" / "cuprite_dataset"
        run_test_set(model, processor, str(test_json), str(data_dir))
    
    elif args.image:
        # Single image inference
        response = inference_single(model, processor, args.image, args.prompt)
        print(f"\nQuestion: {args.prompt}")
        print(f"\nResponse: {response}")
    
    else:
        # Interactive mode
        print("\nInteractive mode. Enter image path and question (or 'quit' to exit):")
        
        while True:
            image_path = input("\nImage path: ").strip()
            if image_path.lower() == 'quit':
                break
            
            if not Path(image_path).exists():
                print(f"File not found: {image_path}")
                continue
            
            prompt = input("Question: ").strip()
            if not prompt:
                prompt = args.prompt
            
            response = inference_single(model, processor, image_path, prompt)
            print(f"\nResponse: {response}")


if __name__ == "__main__":
    main()

