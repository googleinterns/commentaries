# MIT License
#
# Copyright (c) 2018 Jonathan Lorraine

import numpy as np
import scipy

import torch
from torch.autograd import grad
from torch.autograd import Variable


def hessian_vector_product(x_grad, vector, model, k):
    v = Variable(vector)
    z = x_grad @ v
    result = grad(z, model.parameters(), retain_graph=True)
    A_x = gather_flat_grad(result)
    A_x = A_x.view(A_x.size(0), 1)
    A_x = A_x + k * v
    return A_x.data


def conjugate_gradiant(A_grad, b, model, is_cuda, hessian=None):
    x = torch.DoubleTensor(np.zeros((b.size(0), b.size(1))) + 1)
    b_norm = max(b.norm(), 1e-20)
    k = 0
    if hessian is not None:
        I = torch.DoubleTensor(np.identity(hessian.data.size(0)) * 1e-4)
        if is_cuda:
            I = I.cuda()
        hessian += I
    if is_cuda:
        x = x.cuda()
    if hessian is not None:
        r_i = b - hessian @ x
    else:
        r_i = b - hessian_vector_product(A_grad, x, model, k)
    d_i = r_i.clone()
    tolerance = 1e-3
    i = 0
    no_update_i = 0
    max_iter = 5000
    print('i:{}, r norm:{}'.format(i, r_i.norm()))
    min_r = (b - hessian_vector_product(A_grad, x, model, 0)).norm()
    best_x = x.clone()
    while min_r / b_norm > tolerance:
        rr_i = r_i.permute(1, 0) @ r_i
        if hessian is not None:
            A_d = hessian @ d_i
        else:
            A_d = hessian_vector_product(A_grad, d_i, model, k)
        alpha = rr_i / (d_i.permute(1, 0) @ A_d)
        x += alpha * d_i
        r_i -= alpha * A_d
        rr_i_new = r_i.permute(1, 0) @ r_i
        d_i = r_i + (rr_i_new / rr_i) * d_i
        i += 1
        no_update_i += 1
        if rr_i_new.sqrt() < min_r:
            r = b - hessian_vector_product(A_grad, x, model, 0)
            if r.norm() < min_r:
                min_r = r.norm()
                best_x = x.clone()
                no_update_i = 0
        if i % 100 == 0:
            print('i:{}, r norm:{}, min r norm:{}, releative: {} same best: {}'.format(i, rr_i_new.sqrt(), min_r,
                                                                                       min_r / b_norm, no_update_i))
        if i == max_iter or no_update_i == max_iter / 10:
            if min_r / b_norm > 1:
                if max_iter > 10000:
                    break
                max_iter += 1000
                k = b_norm * 4 * 1e-3
                x = best_x.clone()
                r_i = b - hessian_vector_product(A_grad, x, model, k)
                d_i = r_i.clone()
            else:
                break

    r = b - hessian_vector_product(A_grad, best_x, model, 0)
    print(r.norm())
    print(b_norm)
    best_x = best_x.cpu()
    return best_x, i


def preconditioned_conjugate_gradiant(A_grad, b, model, is_cuda, hessian=None):
    x = torch.DoubleTensor(np.random.normal(size=(b.size(0), b.size(1))))
    if hessian is not None:
        I = torch.DoubleTensor(np.identity(hessian.data.size(0)) * 1e-4)
        if is_cuda:
            I = I.cuda()
        hessian += I
    M = scipy.sparse.linalg.spilu(hessian)
    M = torch.DoubleTensor(M)
    if is_cuda:
        x, M = x.cuda(), M.cuda()
    if hessian is not None:
        r_i = b - hessian @ x
    else:
        r_i = b - hessian_vector_product(A_grad, x, model)
    z_i = M @ r_i
    d_i = z_i.clone()
    tolerance = max(1e-4 * b.norm(), 1e-4)
    i = 0
    min_r = r_i.norm()
    best_x = x.clone()
    b_norm = b.norm()
    no_update_i = 0
    max_iter = 10000
    print('i:{}, r norm:{}'.format(i, r_i.norm()))
    while r_i.norm() > tolerance:
        if hessian is not None:
            A_d = hessian @ d_i
        else:
            A_d = hessian_vector_product(A_grad, d_i, model)
        rz_i = r_i.permute(1, 0) @ z_i
        alpha = rz_i / (d_i.permute(1, 0) @ A_d)
        x += alpha * d_i
        r_i -= alpha * A_d
        z_i = M @ r_i
        d_i = z_i + ((z_i.permute(1, 0) @ r_i) / rz_i) * d_i
        i += 1
        no_update_i += 1
        if r_i.norm() < min_r:
            min_r = r_i.norm()
            best_x = x.clone()
            no_update_i = 0
        if i % 10 == 0:
            print('i:{}, r norm:{}, min r norm:{}, releative: {} same best: {}'.format(i, r_i.norm(), min_r / M.norm(),
                                                                                       min_r / M.norm() / b_norm,
                                                                                       no_update_i))
        if i == max_iter:
            break
    print('iteration: {}; result r_i : {}'.format(i, min_r))
    return best_x, i


def test_cg(size):
    A = np.random.randn(size, size)
    A_i = torch.DoubleTensor(np.dot(A, A.T))
    b = torch.DoubleTensor(np.random.randn(size, 1))
    # x, i = conjugate_gradiant(None, b, None, False, A_i)
    x_p, i = preconditioned_conjugate_gradiant(None, b, None, False, A_i)
    x_refer = np.linalg.solve(np.dot(A, A.T), b)
    print("difference between own CG and actual solution")
    # print(np.linalg.norm(x.numpy() - x_refer))
    print("difference between own CG and actual solution")
    print(np.linalg.norm(x_p.numpy() - x_refer))


def gather_flat_grad(loss_grad):
    #cnt = 0
    #for g in loss_grad:
    #    g_vector = g.contiguous().view(-1) if cnt == 0 else torch.cat([g_vector, g.contiguous().view(-1)])
    #    cnt = 1
    return torch.cat([p.reshape(-1) for p in loss_grad]) #g_vector


def eval_hessian(g_vector, model, is_cuda):
    l = g_vector.size(0)
    hessian = torch.zeros(l, l)
    if is_cuda:
        hessian = hessian.cuda()
    for idx in range(l):
        grad2rd = grad(g_vector[idx], model.parameters(), retain_graph=True, allow_unused=True)
        cnt = 0
        for g in grad2rd:
            g2 = g.contiguous().view(-1) if cnt == 0 else torch.cat([g2, g.contiguous().view(-1)])
            cnt = 1
        hessian[idx] = g2

    return hessian


def test_hessian():
    x = Variable(torch.ones(2, 1), requires_grad=True)
    A = Variable(torch.FloatTensor([[1, 2], [3, 4]]), requires_grad=False)
    f = x.view(-1) @ A @ x
    fg = grad(f, x, create_graph=True)
    v = Variable(torch.ones(2, 1), requires_grad=True)
    f_fg = gather_flat_grad(fg)
    print(A + A.t())


def eval_jacobian(lambd_vector, model, is_cuda):
    row_size = lambd_vector.size(0)
    col_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    jacobian = torch.zeros(row_size, col_size)
    if is_cuda: jacobian = jacobian.cuda()
    for idx in range(row_size):
        if idx % 1000 == 0: print(f"jac id {idx} / {row_size}")
        grad2rd = grad(lambd_vector[idx], model.parameters(), retain_graph=True, allow_unused=True)
        cnt = 0
        for g in grad2rd:
            g2 = g.contiguous().view(-1) if cnt == 0 else torch.cat([g2, g.contiguous().view(-1)])
            cnt = 1
        jacobian[idx] = g2
    jacobian = jacobian.permute(1, 0)
    return jacobian


def eval_jacobian_matrix(lambd_vector, model, is_cuda):
    size = lambd_vector.size(0)
    jacobian = {}
    for idx in range(size):
        print(idx)
        jacobian[idx] = {}
        for m in model.modules():
            if m.__class__.__name__ in ['Linear', 'Conv2d']:
                grad2d = grad(lambd_vector[idx], m.weight, retain_graph=True, allow_unused=True)[0]

                if m.bias is not None:
                    grad2d = torch.cat(
                        [grad2d, grad(lambd_vector[idx], m.bias, retain_graph=True, allow_unused=True)[0].view(-1, 1)],
                        1)
                jacobian[idx][m] = grad2d
    return jacobian


def test_jacobian():
    x = Variable(torch.ones(2, 1), requires_grad=True)
    y = Variable(torch.FloatTensor([[10], [4]]), requires_grad=True)
    A = Variable(torch.FloatTensor([[1, 2], [3, 4]]), requires_grad=False)
    f = y.view(-1) @ A @ x
    fg = grad(f, x, create_graph=True)
    print(eval_jacobian(gather_flat_grad(fg), y))
    print(A)
