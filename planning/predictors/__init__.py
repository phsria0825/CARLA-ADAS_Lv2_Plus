"""
Motion predictor 서브 패키지.

다른 차량의 미래 trajectory 를 예측해 PredictionSet 형태로 노출한다.
현재는 등속 가정의 ConstantVelocityPredictor 만 제공하며, 추후 CA / CTRV /
학습 기반 예측기를 같은 인터페이스 (``predict(fused, t_now, lane) ->
PredictionSet``) 로 확장할 수 있다.
"""
from planning.predictors.constant_velocity import ConstantVelocityPredictor

__all__ = ["ConstantVelocityPredictor"]
