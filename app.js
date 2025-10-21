class PresentationApp {
    constructor() {
        this.currentSlide = 1;
        this.totalSlides = 12;
        this.init();
    }

    init() {
        this.bindEvents();
        this.updateUI();
        this.updateProgressBar();
    }

    bindEvents() {
        // Navigation buttons
        const prevBtn = document.getElementById('prevBtn');
        const nextBtn = document.getElementById('nextBtn');

        prevBtn.addEventListener('click', () => this.prevSlide());
        nextBtn.addEventListener('click', () => this.nextSlide());

        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            switch(e.key) {
                case 'ArrowLeft':
                case 'ArrowUp':
                    e.preventDefault();
                    this.prevSlide();
                    break;
                case 'ArrowRight':
                case 'ArrowDown':
                case ' ': // Spacebar
                    e.preventDefault();
                    this.nextSlide();
                    break;
                case 'Home':
                    e.preventDefault();
                    this.goToSlide(1);
                    break;
                case 'End':
                    e.preventDefault();
                    this.goToSlide(this.totalSlides);
                    break;
                case 'Escape':
                    e.preventDefault();
                    this.toggleFullscreen();
                    break;
                default:
                    // Check for number keys (1-9, 0)
                    if (e.key >= '0' && e.key <= '9') {
                        const slideNum = parseInt(e.key);
                        if (slideNum >= 1 && slideNum <= this.totalSlides) {
                            e.preventDefault();
                            this.goToSlide(slideNum);
                        }
                    }
                    break;
            }
        });

        // Touch/swipe support for mobile
        let startX = 0;
        let endX = 0;
        const slidesContainer = document.getElementById('slidesContainer');

        slidesContainer.addEventListener('touchstart', (e) => {
            startX = e.touches[0].clientX;
        }, { passive: true });

        slidesContainer.addEventListener('touchend', (e) => {
            endX = e.changedTouches[0].clientX;
            this.handleSwipe(startX, endX);
        }, { passive: true });

        // Mouse wheel support
        document.addEventListener('wheel', (e) => {
            if (Math.abs(e.deltaY) > 50) { // Threshold to prevent accidental scrolling
                e.preventDefault();
                if (e.deltaY > 0) {
                    this.nextSlide();
                } else {
                    this.prevSlide();
                }
            }
        }, { passive: false });

        // Window resize handler
        window.addEventListener('resize', () => {
            this.handleResize();
        });

        // Presentation mode toggle (F11)
        document.addEventListener('keydown', (e) => {
            if (e.key === 'F11') {
                e.preventDefault();
                this.toggleFullscreen();
            }
        });
    }

    handleSwipe(startX, endX) {
        const diffX = startX - endX;
        const threshold = 50; // Minimum swipe distance

        if (Math.abs(diffX) > threshold) {
            if (diffX > 0) {
                // Swiped left - next slide
                this.nextSlide();
            } else {
                // Swiped right - previous slide
                this.prevSlide();
            }
        }
    }

    prevSlide() {
        if (this.currentSlide > 1) {
            this.goToSlide(this.currentSlide - 1);
        }
    }

    nextSlide() {
        if (this.currentSlide < this.totalSlides) {
            this.goToSlide(this.currentSlide + 1);
        }
    }

    goToSlide(slideNumber) {
        if (slideNumber < 1 || slideNumber > this.totalSlides) {
            return;
        }

        // Remove active class from current slide
        const currentSlideElement = document.querySelector('.slide.active');
        if (currentSlideElement) {
            currentSlideElement.classList.remove('active');
            // Add exit animation class
            if (slideNumber > this.currentSlide) {
                currentSlideElement.classList.add('prev');
            } else {
                currentSlideElement.classList.add('next');
            }
        }

        // Update current slide
        this.currentSlide = slideNumber;

        // Add active class to new slide
        const newSlideElement = document.querySelector(`[data-slide="${slideNumber}"]`);
        if (newSlideElement) {
            setTimeout(() => {
                // Remove all transition classes
                document.querySelectorAll('.slide').forEach(slide => {
                    slide.classList.remove('prev', 'next');
                });
                
                newSlideElement.classList.add('active');
                this.updateUI();
                this.updateProgressBar();
                this.announceSlideChange();
            }, 150); // Slight delay for smooth transition
        }
    }

    updateUI() {
        // Update slide counter
        document.getElementById('currentSlide').textContent = this.currentSlide;
        document.getElementById('totalSlides').textContent = this.totalSlides;

        // Update navigation buttons
        const prevBtn = document.getElementById('prevBtn');
        const nextBtn = document.getElementById('nextBtn');

        prevBtn.disabled = this.currentSlide === 1;
        nextBtn.disabled = this.currentSlide === this.totalSlides;

        // Update page title
        const slideTitle = document.querySelector(`[data-slide="${this.currentSlide}"] h1`)?.textContent;
        if (slideTitle) {
            document.title = `${slideTitle} - ML Penguin Presentation`;
        }
    }

    updateProgressBar() {
        const progressBar = document.getElementById('progressBar');
        const progress = (this.currentSlide / this.totalSlides) * 100;
        progressBar.style.width = `${progress}%`;
    }

    announceSlideChange() {
        // Accessibility: Announce slide change for screen readers
        const announcement = `Slide ${this.currentSlide} of ${this.totalSlides}`;
        const ariaLive = document.createElement('div');
        ariaLive.setAttribute('aria-live', 'polite');
        ariaLive.setAttribute('aria-atomic', 'true');
        ariaLive.classList.add('sr-only');
        ariaLive.textContent = announcement;
        
        document.body.appendChild(ariaLive);
        
        // Remove after announcement
        setTimeout(() => {
            document.body.removeChild(ariaLive);
        }, 1000);
    }

    handleResize() {
        // Handle responsive adjustments if needed
        // This could include adjusting font sizes, layout, etc.
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;
        
        // Add viewport size classes for CSS targeting
        document.body.classList.remove('viewport-small', 'viewport-medium', 'viewport-large');
        
        if (viewportWidth < 768) {
            document.body.classList.add('viewport-small');
        } else if (viewportWidth < 1200) {
            document.body.classList.add('viewport-medium');
        } else {
            document.body.classList.add('viewport-large');
        }
    }

    toggleFullscreen() {
        if (!document.fullscreenElement) {
            document.documentElement.requestFullscreen().catch(err => {
                console.log('Error attempting to enable fullscreen:', err);
            });
        } else {
            document.exitFullscreen();
        }
    }

    // Public method to jump to a specific slide (useful for debugging or direct access)
    jumpToSlide(slideNumber) {
        this.goToSlide(slideNumber);
    }

    // Method to get current presentation state
    getState() {
        return {
            currentSlide: this.currentSlide,
            totalSlides: this.totalSlides,
            progress: (this.currentSlide / this.totalSlides) * 100
        };
    }

    // Method to add custom keyboard shortcuts
    addKeyboardShortcut(key, callback) {
        document.addEventListener('keydown', (e) => {
            if (e.key === key) {
                e.preventDefault();
                callback(this);
            }
        });
    }
}

// Screen reader support class for better accessibility
class AccessibilitySupport {
    constructor() {
        this.initAccessibilityFeatures();
    }

    initAccessibilityFeatures() {
        // Add skip navigation link
        this.addSkipNavigation();
        
        // Enhance focus management
        this.enhanceFocusManagement();
        
        // Add ARIA landmarks
        this.addARIALandmarks();
    }

    addSkipNavigation() {
        const skipLink = document.createElement('a');
        skipLink.href = '#main-content';
        skipLink.textContent = 'Skip to main content';
        skipLink.classList.add('sr-only-focusable');
        skipLink.style.cssText = `
            position: absolute;
            top: -40px;
            left: 6px;
            z-index: 10000;
            padding: 8px;
            background: var(--color-primary);
            color: white;
            text-decoration: none;
            border-radius: 4px;
        `;
        
        skipLink.addEventListener('focus', () => {
            skipLink.style.top = '6px';
        });
        
        skipLink.addEventListener('blur', () => {
            skipLink.style.top = '-40px';
        });
        
        document.body.insertBefore(skipLink, document.body.firstChild);
    }

    enhanceFocusManagement() {
        // Ensure slides are focusable for keyboard users
        document.querySelectorAll('.slide').forEach(slide => {
            slide.setAttribute('tabindex', '-1');
        });
        
        // Focus management for slide transitions
        const originalGoToSlide = PresentationApp.prototype.goToSlide;
        PresentationApp.prototype.goToSlide = function(slideNumber) {
            originalGoToSlide.call(this, slideNumber);
            
            setTimeout(() => {
                const activeSlide = document.querySelector('.slide.active');
                if (activeSlide) {
                    activeSlide.focus();
                }
            }, 200);
        };
    }

    addARIALandmarks() {
        // Add main content area
        const slidesContainer = document.getElementById('slidesContainer');
        slidesContainer.setAttribute('role', 'main');
        slidesContainer.setAttribute('id', 'main-content');
        slidesContainer.setAttribute('aria-label', 'Presentation slides');
        
        // Add navigation landmark
        const navControls = document.querySelector('.nav-controls');
        navControls.setAttribute('role', 'navigation');
        navControls.setAttribute('aria-label', 'Slide navigation');
        
        // Add live region for slide announcements
        const liveRegion = document.createElement('div');
        liveRegion.setAttribute('aria-live', 'polite');
        liveRegion.setAttribute('aria-atomic', 'true');
        liveRegion.classList.add('sr-only');
        liveRegion.id = 'slide-announcements';
        document.body.appendChild(liveRegion);
    }
}

// Initialize the presentation when the page loads
document.addEventListener('DOMContentLoaded', () => {
    // Initialize main presentation app
    const presentationApp = new PresentationApp();
    
    // Initialize accessibility support
    const accessibilitySupport = new AccessibilitySupport();
    
    // Make the app globally accessible for debugging
    window.presentationApp = presentationApp;
    
    // Add some helpful keyboard shortcuts
    presentationApp.addKeyboardShortcut('r', (app) => {
        // Reset to first slide
        app.goToSlide(1);
    });
    
    presentationApp.addKeyboardShortcut('e', (app) => {
        // Go to last slide
        app.goToSlide(app.totalSlides);
    });
    
    // Log helpful information
    console.log('🐧 Penguin ML Presentation loaded successfully!');
    console.log('Navigation: Arrow keys, Space, Home/End, Number keys (1-9)');
    console.log('Shortcuts: R (restart), E (end), F11 (fullscreen), Esc (exit fullscreen)');
    console.log('Touch: Swipe left/right on mobile devices');
    console.log('Mouse: Scroll wheel to navigate');
    
    // Show initial loading message
    setTimeout(() => {
        const announcement = 'Presentation ready. Use arrow keys or buttons to navigate between slides.';
        const ariaLive = document.createElement('div');
        ariaLive.setAttribute('aria-live', 'polite');
        ariaLive.classList.add('sr-only');
        ariaLive.textContent = announcement;
        document.body.appendChild(ariaLive);
        
        setTimeout(() => {
            if (document.body.contains(ariaLive)) {
                document.body.removeChild(ariaLive);
            }
        }, 3000);
    }, 1000);
});

// Utility functions for presentation control
function goToSlide(slideNumber) {
    if (window.presentationApp) {
        window.presentationApp.jumpToSlide(slideNumber);
    }
}

function getCurrentSlide() {
    return window.presentationApp ? window.presentationApp.getState() : null;
}

// Export for potential module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { PresentationApp, AccessibilitySupport };
}