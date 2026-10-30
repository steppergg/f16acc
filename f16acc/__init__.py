"""Near-FP32-accurate FP16 GEMM at full tensor rate on stock GeForce cards.

GeForce runs FP16 tensor math with FP32 accumulate at half rate, but with FP16
accumulate at full rate. f16acc accumulates each 32-long K chunk in FP16 and
adds the chunks in FP32, so rounding error stays near FP32-accumulate levels.
"""

from .kernel import mm
from .patch import calls, disable, enable

__all__ = ["mm", "enable", "disable", "calls"]
