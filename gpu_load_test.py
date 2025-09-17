"""
Simple GPU load generator to test monitoring
"""
import numpy as np
import time

def generate_gpu_load():
    """Generate some computational load to test GPU monitoring"""
    print("🔥 Generating CPU/GPU load for testing...")
    print("   (This will make your fans spin up!)")
    
    # Generate large matrices for computation
    size = 5000
    
    for i in range(10):
        print(f"   Load iteration {i+1}/10...")
        
        # Create large random matrices
        a = np.random.rand(size, size).astype(np.float32)
        b = np.random.rand(size, size).astype(np.float32)
        
        # Perform matrix multiplication (CPU intensive)
        c = np.dot(a, b)
        
        # Add some additional operations
        d = np.sin(c) + np.cos(c)
        result = np.sum(d)
        
        print(f"   Result sum: {result:.2e}")
        time.sleep(1)
    
    print("✅ Load test completed!")

if __name__ == "__main__":
    generate_gpu_load()



