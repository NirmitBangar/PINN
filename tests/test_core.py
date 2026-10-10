import torch
from src.pinn import OscillatorConfig, exact_solution, PositiveParameter, FourierPINN

def test_initial_condition():
    cfg=OscillatorConfig(mu=4,k=400)
    t=torch.zeros((1,1))
    assert abs(exact_solution(t,cfg).item()-1.0) < 1e-6

def test_positive_parameter():
    p=PositiveParameter(2.0)
    assert p().item() > 0

def test_fourier_model_shape():
    model=FourierPINN()
    y=model(torch.rand(8,1))
    assert y.shape == (8,1)
