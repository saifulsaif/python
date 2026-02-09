import sys
import platform

def check_env():
    print("--- 🛠️ Python Environment Check ---")
    print(f"Python Version: {sys.version}")
    print(f"Operating System: {platform.system()} {platform.release()}")
    print("\n✅ You are ready to go!")
    print("Follow the ROADMAP.md and start practicing in the 01_basics folder.")

if __name__ == "__main__":
    check_env()
