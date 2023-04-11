from rest_framework import generics
from rest_framework.permissions import IsAuthenticated
from .serializers import SP500PredictionSerializer
from .utils import predictor,update_stock_data


class SP500PricePredictionView1(generics.RetrieveAPIView):
    permission_classes = (IsAuthenticated,)
    serializer_class = SP500PredictionSerializer

    def get_object(self):
        prediction = predictor()
        return {"prediction": prediction}

class SP500PricePredictionView(generics.RetrieveAPIView):
    permission_classes = (IsAuthenticated,)
    serializer_class = SP500PredictionSerializer

    def get_object(self):
        prediction = predictor()
        # 更新数据集
        update_stock_data()  # 新增此行，这个函数在下面定义
        return {"prediction": prediction}

