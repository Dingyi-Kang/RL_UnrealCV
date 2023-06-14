This version fixed the bug of storing reshaped stack frame in buffer(the stored stack frame should not include batch dimension), added notes of why the run out of allocated memory can happen in first CNN layer (200, 32, 4, 336, 336), and why mini-batch is needed

The used mini-batch size is 50
