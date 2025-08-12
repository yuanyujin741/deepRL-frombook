# 学习总结

# 代码书写

## 关于奖励的设计问题：

    在实际代码中，你应该存储**五元组(s, a, r, s', done)**到经验回放缓冲区，而在计算TD目标时：

```python
# 计算TD目标
if done:
    target = reward  # 不考虑未来奖励
else:
    target = reward + gamma * np.max(q_values_next)
```

# 代码调试


# 剩余问题
