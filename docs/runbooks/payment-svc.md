# Runbook: payment-svc High Error Rate

**Service**: payment-svc  
**Namespace**: production  
**Severity**: P1  
**Owner**: Platform Engineering

---

## Symptoms

- HTTP 500 error rate > 5% on `payment-svc`
- Anomaly score > 0.75 from KOGNOS GNN
- OOMKill events in pod events (`kubectl describe pod payment-svc-*`)
- Latency p99 > 2000ms
- Cart abandonment spike in business metrics

## Root Causes

### 1. OOMKill Loop (Most Common)
The payment processor's JVM heap is exhausted under load. The pod is killed
and restarted before it can complete in-flight requests. This creates a
thundering herd — retries increase load on a pod that's already struggling.

**Indicators**:
```
Events:
  Warning  OOMKilling  kubelet  Memory cgroup out of memory
```

**Resolution**:
```bash
# Immediate: scale out to distribute load
kubectl scale deployment/payment-svc --replicas=5 -n production

# Verify pods are ready
kubectl get pods -l app=payment-svc -n production -w

# Check memory usage (should stabilise below 85%)
kubectl top pod -l app=payment-svc -n production
```

### 2. Downstream Dependency Failure
`payment-svc` depends on `postgres-payment`. If the database is slow or
unavailable, payment requests time out and return 500s.

**Check**:
```bash
kubectl describe pod -l app=postgres-payment -n data
kubectl logs -l app=postgres-payment -n data --tail=50
```

**Resolution**: If Postgres is OOMKilling, increase memory limits or scale vertically.

### 3. Fraud Service Latency
`payment-svc → fraud-svc` calls block payment processing. If `fraud-svc`
is slow, payments queue up and time out.

**Check**:
```bash
# Look for high latency on fraud-svc
kubectl top pod -l app=fraud-svc -n production
```

## Prevention

- Set JVM heap limits explicitly: `-Xms512m -Xmx1g` (match container limit)
- Add circuit breaker (Resilience4j) on payment → fraud-svc calls
- HPA: auto-scale payment-svc when CPU > 70%
- Alert on OOMKill events (Prometheus `kube_pod_container_status_restarts_total`)

## Escalation

If scaling doesn't resolve within 10 minutes:
1. Page the payments team on-call (#payments-oncall Slack)
2. Activate DR: route traffic to backup payment processor
3. Open incident in PagerDuty with tag `payment-p1`

## Related Incidents

- INC-2024-11-03: OOMKill cascade during Black Friday (45 min outage)
- INC-2024-09-17: Fraud service latency caused payment cascade
