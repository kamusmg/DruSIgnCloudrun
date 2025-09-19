// Fix: Implemented state persistence utility functions.

const STATE_KEY = 'facadeAppState';

/**
 * Saves the application state to localStorage.
 * @param state The state object to save.
 */
export const saveState = <T,>(state: T): void => {
  try {
    const serializedState = JSON.stringify(state);
    localStorage.setItem(STATE_KEY, serializedState);
  } catch (error) {
    console.warn("Could not save app state:", error);
  }
};

/**
 * Loads the application state from localStorage.
 * @returns The saved state object or null if it doesn't exist or fails to parse.
 */
export const loadState = <T,>(): T | null => {
  try {
    const serializedState = localStorage.getItem(STATE_KEY);
    if (serializedState === null) {
      return null;
    }
    return JSON.parse(serializedState) as T;
  } catch (error) {
    console.warn("Could not load app state:", error);
    return null;
  }
};

/**
 * Clears the saved application state from localStorage.
 */
export const clearState = (): void => {
    try {
        localStorage.removeItem(STATE_KEY);
    } catch (error) {
        console.warn("Could not clear app state:", error);
    }
}
