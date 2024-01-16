# tests/test_condense_triples.py
import torch
from src.rao_tools import condense_triples, compute_cumulative_averages

def test_condense_triples_single_triple():
    rao_triple = (torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]]), torch.tensor([[5.0, 6.0]]))
    expected = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    result = condense_triples([rao_triple])
    assert torch.equal(result, expected), f"Expected {expected}, got {result}"

def test_condense_triples_multiple_triples():
    rao_triples = [(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]]), torch.tensor([[5.0, 6.0]])),
                   (torch.tensor([[7.0, 8.0]]), torch.tensor([[9.0, 10.0]]), torch.tensor([[11.0, 12.0]]))]
    expected = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]])
    result = condense_triples(rao_triples)
    assert torch.equal(result, expected), f"Expected {expected}, got {result}"

def test_condense_triples_single_triple_multiple_batches():
    rao_triple = (torch.tensor([[[1.0, 2.0]], [[7.0, 8.0]]]), 
                  torch.tensor([[[3.0, 4.0]], [[9.0, 10.0]]]), 
                  torch.tensor([[[5.0, 6.0]], [[11.0, 12.0]]]))
    expected = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], 
                             [[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]])
    result = condense_triples([rao_triple])
    assert torch.equal(result, expected), f"Expected {expected}, got {result}"

def test_condense_triples_multiple_triples_multiple_batches():
    rao_triples = [(torch.tensor([[[1.0, 2.0]], [[7.0, 8.0]]]), 
                    torch.tensor([[[3.0, 4.0]], [[9.0, 10.0]]]), 
                    torch.tensor([[[5.0, 6.0]], [[11.0, 12.0]]])),
                   (torch.tensor([[[13.0, 14.0]], [[19.0, 20.0]]]), 
                    torch.tensor([[[15.0, 16.0]], [[21.0, 22.0]]]), 
                    torch.tensor([[[17.0, 18.0]], [[23.0, 24.0]]]))]
    expected = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]], 
                             [[7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0]]])
    result = condense_triples(rao_triples)
    assert torch.equal(result, expected), f"Expected {expected}, got {result}"

def test_compute_cumulative_averages_single_value():
    losses = torch.tensor([[1.0]])
    expected = torch.tensor([[1.0]])
    result = compute_cumulative_averages(losses)
    assert torch.equal(result, expected), f"Expected {expected}, got {result}"

def test_compute_cumulative_averages_constant_values():
    losses = torch.tensor([[2.0, 2.0, 2.0]])
    expected = torch.tensor([[2.0, 2.0, 2.0]])
    result = compute_cumulative_averages(losses)
    assert torch.equal(result, expected), f"Expected {expected}, got {result}"

def test_compute_cumulative_averages_decreasing_values():
    losses = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    expected = torch.tensor([[2.5, 2.0, 1.5, 1.0]])
    result = compute_cumulative_averages(losses)
    assert torch.allclose(result, expected), f"Expected {expected}, got {result}"

def test_compute_cumulative_averages_increasing_values():
    losses = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    expected = torch.tensor([[2.5, 3.0, 3.5, 4.0]])
    result = compute_cumulative_averages(losses)
    assert torch.allclose(result, expected), f"Expected {expected}, got {result}"

def test_compute_cumulative_averages_multiple_rows():
    losses = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    expected = torch.tensor([[2.0, 2.5, 3.0], [2.0, 1.5, 1.0]])
    result = compute_cumulative_averages(losses)
    assert torch.allclose(result, expected), f"Expected {expected}, got {result}"