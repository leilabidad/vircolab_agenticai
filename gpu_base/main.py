import subprocess

def main():
    options = {
        "0": "run_prepare_metadata.py",
        "1": "run_prepare_dataset.py",
        "1-1": "run_minimize_dataset.py",
        "2": "run_train_swin_cm.py",
        "3": "run_compute_sc.py",
        "4": "run_compute_dcs_weights.py",
        "5": "run_pipeline_example.py"
    }

    print("Select the action to perform:")
    print("0: Prepare metadata")
    print("1: Prepare dataset")
    print("1-1: Minimize dataset")
    print("2: Train SwinNet 2D and compute Cm")
    print("3: Compute Sc")
    print("4: Compute DCS weights")
    print("5: Run pipeline example")
        
    choice = input("Enter your choice (0-5): ").strip()
    script = options.get(choice)
    
    if script:
        subprocess.run(["python", script])
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
