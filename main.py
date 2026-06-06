"""
ORFormer-Lite: Real-Time Facial Expression Recognition for Assistive Communication

Main entry point for the application.
"""

import argparse
import sys
import os


def main():
    parser = argparse.ArgumentParser(
        description="ORFormer-Lite: Facial Expression Recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run webcam demo
  py main.py demo

  # Run with GUI
  py main.py gui

  # Start API server
  py main.py server

  # Train model with synthetic data
  py main.py train --synthetic --samples 200

  # Train model with webcam capture
  py main.py train --mode capture --samples 50

  # Evaluate trained model
  py main.py train --mode evaluate
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run webcam demo")
    demo_parser.add_argument("--model", type=str, default=None, help="Path to trained model")
    demo_parser.add_argument("--camera", type=int, default=0, help="Camera ID")
    demo_parser.add_argument("--no-landmarks", action="store_true", help="Hide landmarks")
    demo_parser.add_argument("--no-expression", action="store_true", help="Hide expression")
    demo_parser.add_argument("--no-intent", action="store_true", help="Hide intent")

    # GUI command
    gui_parser = subparsers.add_parser("gui", help="Run with GUI")

    # Server command
    server_parser = subparsers.add_parser("server", help="Start API server")
    server_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host")
    server_parser.add_argument("--port", type=int, default=8000, help="Port")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument(
        "--mode",
        choices=["capture", "train", "evaluate"],
        default="train",
        help="Training mode",
    )
    train_parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    train_parser.add_argument("--samples", type=int, default=200, help="Samples per class")
    train_parser.add_argument(
        "--model-type",
        choices=["rf", "gb"],
        default="rf",
        help="Model type (rf=RandomForest, gb=GradientBoosting)",
    )
    train_parser.add_argument(
        "--model-path",
        type=str,
        default="checkpoints/expression_model.joblib",
        help="Model save path",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # Ensure directories exist
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    if args.command == "demo":
        from src.demo.webcam_demo import WebcamDemo

        demo = WebcamDemo(
            model_path=args.model,
            camera_id=args.camera,
            show_landmarks=not args.no_landmarks,
            show_expression=not args.no_expression,
            show_intent=not args.no_intent,
        )
        demo.start()

    elif args.command == "gui":
        from src.demo.gui import main as gui_main
        gui_main()

    elif args.command == "server":
        from src.demo.server import run_server
        run_server(host=args.host, port=args.port)

    elif args.command == "train":
        from src.utils.training import (
            ExpressionDatasetBuilder,
            ExpressionTrainer,
        )
        import json

        dataset = ExpressionDatasetBuilder()

        if args.mode == "capture":
            dataset.capture_interactive(samples_per_class=args.samples)
            dataset.save()

        elif args.mode == "train":
            if args.synthetic:
                print("Generating synthetic training data...")
                dataset.generate_synthetic(samples_per_class=args.samples)
                dataset.save()
            else:
                dataset.load()

            trainer = ExpressionTrainer(model_type=args.model_type)
            X, y = trainer.prepare_data(dataset)
            print("\nTraining model...")
            metrics = trainer.train(X, y)

            trainer.save(args.model_path)

            with open("checkpoints/metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

        elif args.mode == "evaluate":
            dataset.load()
            trainer = ExpressionTrainer(model_type=args.model_type)
            trainer.load(args.model_path)
            X, y = trainer.prepare_data(dataset)
            metrics = trainer.train(X, y)


if __name__ == "__main__":
    main()
