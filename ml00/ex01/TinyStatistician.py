from typing import Union

class TinyStatistician:

    @staticmethod
    def mean(x) -> Union[float, None]:
        if len(x) == 0:
            return None

        return sum(x) / len(x)

    @staticmethod
    def median(x) -> Union[float, None]:
        if len(x) == 0:
            return None

        x.sort()
        count = len(x)
        if count % 2 == 0:
            return (float(x[int(count // 2)]) + float(x[int(count // 2) - 1])) / 2

        return float(x[count // 2])

    @staticmethod
    def quartile(x) -> Union[float, None]:
        if len(x) == 0:
            return None

        x.sort()
        count = len(x)
        return [float(x[count // 4]), float(x[count * 3 // 4])]

    @staticmethod
    def percentile(x, p) -> Union[float, None]:
        if not x:
            return None

        if not 0 <= p <= 100:
            return None

        data_sorted = sorted(x)
        index = (p / 100) * (len(data_sorted) - 1)
        lower_index = int(index // 1)
        upper_index = lower_index + 1
        lower_value = data_sorted[lower_index]
        upper_value = data_sorted[upper_index]

        return (1 - (index - lower_index)) * lower_value + (index - lower_index) * upper_value

    @staticmethod
    def var(x) -> Union[float, None]:
        if len(x) == 0:
            return None

        mean = TinyStatistician.mean(x)
        return sum([(i - mean) ** 2 for i in x]) / (len(x) - 1)

    @staticmethod
    def std(x) -> Union[float, None]:
        variance = TinyStatistician.var(x)

        if variance is None:
            return None

        return variance ** 0.5