from nndiy.linalg import Vector, Number


class CostFunction:
    def calc(self, predicted: Vector, actual: Vector) -> Number:
        pass

    def jacobian_calc(self, predicted: Vector, actual: Vector) -> Vector:
        pass


class QuadraticDistance(CostFunction):
    def calc(self, predicted: Vector, actual: Vector) -> Number:
        assert predicted.size == actual.size
        return sum([0.5*(predicted.vector[i] - actual.vector[i])**2 for i in range(predicted.size)])

    def jacobian_calc(self, predicted: Vector, actual: Vector) -> Vector:
        return Vector([predicted.vector[i] - actual.vector[i] for i in range(predicted.size)])
