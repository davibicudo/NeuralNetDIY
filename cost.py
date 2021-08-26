from linalg import Vector, Number


class CostFunction:
    def calc(self, predicted: Vector, actual: Vector) -> Number:
        pass


class SquaredDistance(CostFunction):
    def calc(self, predicted: Vector, actual: Vector) -> Number:
        assert predicted.size == actual.size
        return sum([(predicted.vector[i] - actual.vector[i])**2 for i in range(predicted.size)])
