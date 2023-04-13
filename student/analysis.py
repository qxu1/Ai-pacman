"""
Analysis question.
Change these default values to obtain the specified policies through value iteration.
If any question is not possible, return just the constant NOT_POSSIBLE:
```
return NOT_POSSIBLE
```
"""

NOT_POSSIBLE = None


def question2():
    """
    [I make answerNoise to 0 so the agent will get the best value.]
    """

    answerDiscount = 0.9
    answerNoise = 0.0

    return answerDiscount, answerNoise


def question3a():
    """
    [I make the discount rate to 0.9 and noise to 0.2 so the agent will risk
    safe path to exit.]
    """

    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = -2.0

    return answerDiscount, answerNoise, answerLivingReward


def question3b():
    """
    [lower the discount value to make the path more safety.]
    """

    answerDiscount = 0.2
    answerNoise = 0.2
    answerLivingReward = -2.0

    return answerDiscount, answerNoise, answerLivingReward


def question3c():
    """
    [I put 0 on Noise so the agent will get the best value.]
    """

    answerDiscount = 0.9
    answerNoise = 0.0
    answerLivingReward = 0.0

    return answerDiscount, answerNoise, answerLivingReward


def question3d():
    """
    [Enter a description of what you did here.]
    """

    answerDiscount = 0.5
    answerNoise = 0.2
    answerLivingReward = 0.0

    return answerDiscount, answerNoise, answerLivingReward


def question3e():
    """
    [Enter a description of what you did here.]
    """
    answerDiscount = 0.9
    answerNoise = 0.5
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward


def question6():
    """
    [Enter a description of what you did here.]
    """

    answerEpsilon = 0.3
    answerLearningRate = 0.5

    return answerEpsilon, answerLearningRate


if __name__ == '__main__':
    questions = [
        question2,
        question3a,
        question3b,
        question3c,
        question3d,
        question3e,
        question6,
    ]

    print('Answers to analysis questions:')
    for question in questions:
        response = question()
        print('    Question %-10s:\t%s' % (question.__name__, str(response)))
