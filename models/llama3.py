from models.base import register
from models.llama2 import Llama2


@register("meta-llama/Meta-Llama-3")
@register("meta-llama/Llama-3")
class Llama3(Llama2):
    pass
