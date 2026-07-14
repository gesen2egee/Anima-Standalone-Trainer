function createQueueCoordinator(options = {}) {
    const {
        isEnabled,
        getRunningJob,
        getNextJob,
        startJob,
        onQueueEmpty = () => {},
        onError = () => {},
        onTransition = () => {},
        retryDelayMs = 2000,
        setTimer = setTimeout,
        clearTimer = clearTimeout
    } = options;

    if (typeof isEnabled !== 'function') throw new Error('isEnabled is required');
    if (typeof getRunningJob !== 'function') throw new Error('getRunningJob is required');
    if (typeof getNextJob !== 'function') throw new Error('getNextJob is required');
    if (typeof startJob !== 'function') throw new Error('startJob is required');

    let scheduledTimer = null;
    let advanceInFlight = false;

    function cancel() {
        if (!scheduledTimer) return;
        clearTimer(scheduledTimer);
        scheduledTimer = null;
        onTransition({ type: 'cancelled' });
    }

    function requestAdvance(delayMs = 0) {
        if (!isEnabled() || scheduledTimer) return false;

        scheduledTimer = setTimer(async () => {
            scheduledTimer = null;
            try {
                await advanceNow();
            } catch (_) {
                // advanceNow already reports the failure through onError.
            }
        }, Math.max(0, Number(delayMs) || 0));
        scheduledTimer?.unref?.();
        onTransition({ type: 'scheduled', delayMs: Math.max(0, Number(delayMs) || 0) });
        return true;
    }

    async function advanceNow() {
        if (!isEnabled()) {
            cancel();
            onTransition({ type: 'disabled' });
            return { status: 'disabled', started: false };
        }
        if (advanceInFlight) {
            requestAdvance(retryDelayMs);
            onTransition({ type: 'busy' });
            return { status: 'busy', started: false };
        }

        cancel();
        advanceInFlight = true;
        try {
            const runningJob = await getRunningJob();
            if (runningJob) {
                requestAdvance(retryDelayMs);
                onTransition({ type: 'waiting', runningJob });
                return { status: 'waiting', started: false, runningJob };
            }

            const nextJob = getNextJob();
            if (!nextJob) {
                onQueueEmpty();
                onTransition({ type: 'empty' });
                return { status: 'empty', started: false };
            }

            onTransition({ type: 'starting', jobName: nextJob });
            const result = await startJob(nextJob);
            onTransition({ type: 'started', jobName: nextJob });
            return { status: 'started', started: true, jobName: nextJob, result };
        } catch (err) {
            onTransition({ type: 'error', error: err.message });
            onError(err);
            throw err;
        } finally {
            advanceInFlight = false;
        }
    }

    function hasScheduledAdvance() {
        return scheduledTimer !== null;
    }

    return {
        advanceNow,
        cancel,
        hasScheduledAdvance,
        requestAdvance
    };
}

module.exports = { createQueueCoordinator };
