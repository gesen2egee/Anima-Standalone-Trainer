const fs = require('fs');
const path = require('path');

function normalizeState(raw) {
    const items = Array.isArray(raw?.items)
        ? raw.items.map(String).filter(Boolean)
        : [];
    const uniqueItems = [];
    for (const item of items) {
        if (!uniqueItems.includes(item)) uniqueItems.push(item);
    }
    const active = raw?.active && uniqueItems.includes(String(raw.active))
        ? String(raw.active)
        : null;
    return {
        items: uniqueItems,
        active,
        updatedAt: raw?.updatedAt || new Date().toISOString()
    };
}

function createJobQueue({ statePath, jobExists }) {
    if (!statePath) throw new Error('statePath is required');
    const exists = typeof jobExists === 'function' ? jobExists : () => true;
    let state = loadState();

    function loadState() {
        if (!fs.existsSync(statePath)) return normalizeState({});
        try {
            return normalizeState(JSON.parse(fs.readFileSync(statePath, 'utf8')));
        } catch (err) {
            return normalizeState({});
        }
    }

    function saveState() {
        state.updatedAt = new Date().toISOString();
        fs.mkdirSync(path.dirname(statePath), { recursive: true });
        fs.writeFileSync(statePath, JSON.stringify(state, null, 2), 'utf8');
    }

    function pruneMissingJobs() {
        const before = state.items.length;
        state.items = state.items.filter(exists);
        if (state.active && !state.items.includes(state.active)) {
            state.active = null;
        }
        if (state.items.length !== before) saveState();
    }

    function getState() {
        pruneMissingJobs();
        return {
            items: [...state.items],
            active: state.active,
            updatedAt: state.updatedAt
        };
    }

    function enqueue(jobName) {
        const name = String(jobName || '').trim();
        if (!name) throw new Error('jobName is required');
        if (!exists(name)) throw new Error('Job not found');
        if (!state.items.includes(name)) {
            state.items.push(name);
            saveState();
        }
        return getState();
    }

    function remove(jobName) {
        const name = String(jobName || '').trim();
        state.items = state.items.filter(item => item !== name);
        if (state.active === name) state.active = null;
        saveState();
        return getState();
    }

    function move(jobName, targetIndex) {
        const name = String(jobName || '').trim();
        const currentIndex = state.items.indexOf(name);
        if (currentIndex === -1) return getState();
        const nextIndex = Math.max(0, Math.min(Number(targetIndex), state.items.length - 1));
        state.items.splice(currentIndex, 1);
        state.items.splice(nextIndex, 0, name);
        saveState();
        return getState();
    }

    function markActive(jobName) {
        const name = String(jobName || '').trim();
        if (!state.items.includes(name)) throw new Error('Job is not queued');
        state.active = name;
        saveState();
        return getState();
    }

    function completeActive(jobName) {
        const name = String(jobName || '').trim();
        if (state.active === name) state.active = null;
        state.items = state.items.filter(item => item !== name);
        saveState();
        return getState();
    }

    function clearActive(jobName) {
        const name = String(jobName || '').trim();
        if (!name || state.active === name) {
            state.active = null;
            saveState();
        }
        return getState();
    }

    function finishQueuedJob(jobName, { success }) {
        const name = String(jobName || '').trim();
        if (!name) return getState();
        if (success) {
            return completeActive(name);
        }
        return clearActive(name);
    }

    function getNext() {
        pruneMissingJobs();
        if (state.active) return null;
        return state.items[0] || null;
    }

    function getPendingJobNames(allJobNames) {
        const queued = new Set(getState().items);
        return allJobNames.filter(name => !queued.has(name));
    }

    return {
        getState,
        enqueue,
        remove,
        move,
        markActive,
        completeActive,
        clearActive,
        finishQueuedJob,
        getNext,
        getPendingJobNames
    };
}

module.exports = { createJobQueue };
