# Runbook: auth-svc Cascade Failure

**Service**: auth-svc  
**Namespace**: production  
**Severity**: P0 (auth outage = full site outage)  
**Owner**: Identity & Access Platform Team

---

## Symptoms

- `auth-svc` returns 503 or times out
- All authenticated endpoints fail cluster-wide (blast radius: 80%+ traffic)
- KOGNOS alert: `api-gateway` and `frontend` show anomaly score > 0.8
- JWT validation errors flooding application logs

## Why Auth Failures Cascade

The cluster dependency chain:
```
auth-svc
  └── api-gateway  (validates JWT on every request)
        ├── product-svc
        ├── cart-svc
        └── payment-svc
              └── (all downstream)
```

A single auth-svc failure blocks every authenticated API call.
auth-svc is the highest-blast-radius service in the cluster.

## Diagnosis Steps

### Step 1 — Confirm auth-svc is the root cause
```bash
kubectl get pods -l app=auth-svc -n production
kubectl describe pod -l app=auth-svc -n production
```

Look for:
- CrashLoopBackOff
- OOMKilled
- Not Ready

### Step 2 — Check postgres-auth (auth DB)
```bash
kubectl get pods -l app=postgres-auth -n data
kubectl logs -l app=postgres-auth -n data --tail=30
```

### Step 3 — Check recent deployments
```bash
kubectl rollout history deployment/auth-svc -n production
```

A bad deployment is the most common cause of sudden auth-svc failures.

## Remediation

### If deployment issue (most likely):
```bash
# Rollback to previous stable version immediately
kubectl rollout undo deployment/auth-svc -n production

# Monitor rollout
kubectl rollout status deployment/auth-svc -n production

# Verify auth is healthy
kubectl logs -l app=auth-svc -n production --tail=20
```

### If OOMKill:
```bash
kubectl scale deployment/auth-svc --replicas=6 -n production
```

### If postgres-auth is down:
```bash
# Check disk space on PVC
kubectl exec -n data \
  $(kubectl get pod -l app=postgres-auth -n data -o name | head -1) \
  -- df -h

# Restart postgres-auth pod (careful: brief auth outage)
kubectl rollout restart deployment/postgres-auth -n data
```

## Circuit Breaker (Temporary Mitigation)

If auth-svc cannot be recovered quickly, enable bypass mode
(only in break-glass emergencies — approval required):
```bash
# Temporarily disable auth verification on api-gateway (DANGEROUS)
kubectl set env deployment/api-gateway AUTH_BYPASS=true -n production
# REVERT IMMEDIATELY after auth-svc recovery:
kubectl set env deployment/api-gateway AUTH_BYPASS- -n production
```

## Post-Incident

1. Root-cause analysis within 24h
2. Add integration test for auth-svc before deploy
3. Add auth-svc to circuit breaker in api-gateway
4. Runbook review in weekly SRE sync

## Related Incidents

- INC-2024-08-22: Bad deployment took down auth for 12 minutes
- INC-2024-06-05: Postgres PVC full caused auth cascade
