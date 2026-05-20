"""
Simple Gradio viewer to compare a predicted PLY vs GT PLY side by side.

Usage:
    python scripts/ply_viewer.py \
        --predicted outputs/dust3r/exp3/frames_40/predicted.ply \
        --gt        data/co3d/teddybear/101_11758_21048/pointcloud.ply \
        --title     "Exp3 - 40 frames (mask 25%)" \
        --port      7861
"""

import argparse
import gradio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predicted", required=True)
    parser.add_argument("--gt",        required=True)
    parser.add_argument("--title",     default="DUSt3R PLY Viewer")
    parser.add_argument("--port",      type=int, default=7861)
    args = parser.parse_args()

    with gradio.Blocks(title=args.title) as demo:
        gradio.HTML(f'<h2 style="text-align:center">{args.title}</h2>')
        with gradio.Row():
            with gradio.Column():
                gradio.HTML('<h3 style="text-align:center">DUSt3R Predicted</h3>')
                gradio.Model3D(value=args.predicted, label="Predicted", clear_color=[0,0,0,1])
            with gradio.Column():
                gradio.HTML('<h3 style="text-align:center">Ground Truth</h3>')
                gradio.Model3D(value=args.gt, label="Ground Truth", clear_color=[0,0,0,1])

    print(f"\n{'='*60}")
    print(f"  Viewer ready!")
    print(f"  Open: http://localhost:{args.port}")
    print(f"{'='*60}\n")

    demo.launch(server_name="0.0.0.0", server_port=args.port, share=False)


if __name__ == "__main__":
    main()
