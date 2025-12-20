import subprocess

def main():
    options = {
        "1": "run_prepare_dataset.py",
        "2": "run_train_swin_cm.py",
        "3": "run_compute_sc.py",
        "4": "run_compute_dcs_weights.py"
    }

    print("Select the action to perform:")
    print("1: Prepare dataset")
    print("2: Train SwinNet 2D and compute Cm")
    print("3: Compute Sc")
    print("4: Compute DCS weights")
    
    choice = input("Enter your choice (1-4): ").strip()
    script = options.get(choice)
    
    if script:
        subprocess.run(["python", script])
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
