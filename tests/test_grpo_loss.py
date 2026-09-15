import pytest
from grpo.loss import group_advantages,token_loss


@pytest.mark.parametrize("rewards",[[0,0,0,0],[1,1,1,1],[0.02]*4])
def test_equal_rewards_zero_advantages(rewards):
    assert group_advantages(rewards) == [0.0]*4


def test_mixed_group_and_std_floor():
    assert group_advantages([0,0,1,1]) == [-1,-1,1,1]
    assert group_advantages([1,1.001]) == pytest.approx([-0.005,0.005])


@pytest.mark.parametrize("rewards",[[0],[float("nan"),0],[float("inf"),0]])
def test_invalid_rewards(rewards):
    with pytest.raises(ValueError): group_advantages(rewards)


def test_tensor_objective_clipping_kl_and_detached_old_reference():
    torch = pytest.importorskip("torch")
    old = torch.tensor([-2.,-3.],requires_grad=True)
    ref = torch.tensor([-2.,-3.],requires_grad=True)
    current = torch.tensor([-2.,-2.],requires_grad=True)
    loss,stats = token_loss(current,old,ref,1.)
    expected = -2.2 + 0.02*(torch.exp(torch.tensor(-1.))-1+1)
    assert loss.item() == pytest.approx(expected.item())
    assert stats["clip_count"] == 1 and stats["token_count"] == 2
    loss.backward()
    assert current.grad is not None and old.grad is None and ref.grad is None


def test_equal_group_initial_reference_has_exactly_zero_gradient():
    torch = pytest.importorskip("torch")
    current = torch.tensor([-2.,-3.],requires_grad=True)
    loss,_ = token_loss(current,current.detach().clone(),current.detach().clone(),0.)
    loss.backward()
    assert loss.item() == 0 and torch.count_nonzero(current.grad).item() == 0


@pytest.mark.parametrize("shape",[[],[1,2,3]])
def test_mismatched_token_shapes(shape):
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError):
        token_loss(torch.tensor(shape,dtype=torch.float32),torch.ones(2),torch.ones(2),1.)
