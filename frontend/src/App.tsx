import { useState, useEffect } from 'react';
import './App.css';
import {
	GITHUB_URL,
	LINKEDIN_URL,
	TWITTER_URL,
	GMAIL_URL,
	FOOTER_TEXT,
	APP_NAME,
	SOMITAV_GITHUB_URL,
	SOMITAV_LINKEDIN_URL,
	SOMITAV_GMAIL_URL,
} from './libs/constants';

interface PredictionResult {
	is_spam: boolean;
	confidence: number;
	details: {
		raw_label: string;
		raw_score: number;
		model_path: string;
		content_length: number;
	};
}

function App() {
	const [emailContent, setEmailContent] = useState('');
	const [characterCount, setCharacterCount] = useState(0);
	const [isLoading, setIsLoading] = useState(false);
	const [result, setResult] = useState<PredictionResult | null>(null);
	const [error, setError] = useState<string | null>(null);
	const [windowWidth, setWindowWidth] = useState(window.innerWidth);

	// Handle window resize for responsive adjustments
	useEffect(() => {
		const handleResize = () => {
			setWindowWidth(window.innerWidth);
		};

		window.addEventListener('resize', handleResize);
		return () => {
			window.removeEventListener('resize', handleResize);
		};
	}, []);

	const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
		const content = e.target.value;
		setEmailContent(content);
		setCharacterCount(content.length);

		// Reset results when input changes
		if (result) {
			setResult(null);
		}
		if (error) {
			setError(null);
		}
	};

	const handleCheckEmail = async () => {
		setIsLoading(true);
		setError(null);
		setResult(null);

		try {
			const response = await fetch(
				import.meta.env.VITE_SERVER_URL + '/predict',
				{
					method: 'POST',
					headers: {
						'Content-Type': 'application/json',
					},
					body: JSON.stringify({ email_content: emailContent }),
				}
			);

			if (!response.ok) {
				throw new Error(`Error: ${response.statusText}`);
			}

			const data = await response.json();
			setResult(data);
		} catch (err) {
			setError(err instanceof Error ? err.message : 'Failed to check email');
			console.error('Error checking email:', err);
		} finally {
			setIsLoading(false);
		}
	};

	const handlePasteSample = () => {
		const sampleEmail = `Subject: URGENT - Your Account Has Been Compromised

Dear Valued Customer,

We have detected suspicious activity on your account. Please click the link below to verify your identity and secure your account immediately:

http://suspicious-link.com/verify

Failure to act within 24 hours will result in permanent account closure.

Security Team`;

		setEmailContent(sampleEmail);
		setCharacterCount(sampleEmail.length);
		setResult(null);
		setError(null);
	};

	const handleClearText = () => {
		setEmailContent('');
		setCharacterCount(0);
		setResult(null);
		setError(null);
	};

	const renderResult = () => {
		if (!result) return null;

		const resultClass = result.is_spam ? 'result spam' : 'result safe';
		const confidencePercent = (result.confidence * 100).toFixed(1);

		return (
			<div className={resultClass}>
				<h3>{result.is_spam ? '⚠️ Spam Detected' : '✅ Email Appears Safe'}</h3>
				<p>
					Confidence: <strong>{confidencePercent}%</strong>
				</p>
				{windowWidth > 480 && (
					<p className='result-detail'>
						{result.is_spam
							? 'This email contains patterns commonly associated with spam or phishing attempts.'
							: 'This email appears to be legitimate based on our analysis.'}
					</p>
				)}
			</div>
		);
	};

	return (
		<div className='container'>
			<h1>{APP_NAME}</h1>
			<p className='description'>
				Paste an email to check if it's spam or safe. Our advanced AI analyzes
				the content for suspicious patterns.
			</p>

			<div className='email-input-container'>
				<textarea
					placeholder='Paste the email content here...'
					value={emailContent}
					onChange={handleInputChange}
					className='email-input'
				/>
				{emailContent && (
					<button
						onClick={handleClearText}
						className='clear-button'
						aria-label='Clear text'
					>
						✕
					</button>
				)}
			</div>

			<div className='counter-row'>
				<span>{characterCount} characters</span>
				<button onClick={handlePasteSample} className='sample-button'>
					<span className='icon'>🔒</span> Paste sample email
				</button>
			</div>

			{error && <div className='error-message'>{error}</div>}
			{isLoading && <div className='loading-message'>Analyzing email...</div>}
			{!isLoading && renderResult()}

			<button
				onClick={handleCheckEmail}
				className='check-button'
				disabled={emailContent.length === 0 || isLoading}
			>
				{isLoading ? 'Analyzing...' : 'Check Email'}
			</button>

			<div className='features'>
				<div className='feature'>
					<div className='feature-icon secure'>🛡️</div>
					<h3>Safe & Private</h3>
					<p>
						Your email content is never stored and analysis happens in
						real-time.
					</p>
				</div>

				<div className='feature'>
					<div className='feature-icon ai'>📈</div>
					<h3>Advanced AI</h3>
					<p>
						Our system uses machine learning to detect even sophisticated
						phishing attempts.
					</p>
				</div>

				<div className='feature'>
					<div className='feature-icon fast'>⚡</div>
					<h3>Instant Results</h3>
					<p>Get immediate analysis with detailed explanations in seconds.</p>
				</div>
			</div>

			<footer>
				<div className='footer-text'>{FOOTER_TEXT}</div>
				<div className='social-links'>
					<div className='contributor'>
						<span className='contributor-name'>Ayush:</span>
						<a
							href={GITHUB_URL}
							target='_blank'
							rel='noopener noreferrer'
							className='social-link'
							title='GitHub'
						>
							<svg viewBox='0 0 24 24' className='icon github-icon'>
								<path d='M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z' />
							</svg>
						</a>
						<a
							href={LINKEDIN_URL}
							target='_blank'
							rel='noopener noreferrer'
							className='social-link'
							title='LinkedIn'
						>
							<svg viewBox='0 0 24 24' className='icon linkedin-icon'>
								<path d='M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z' />
							</svg>
						</a>
						<a
							href={`mailto:${GMAIL_URL}`}
							className='social-link'
							title='Email'
						>
							<svg viewBox='0 0 24 24' className='icon email-icon'>
								<path d='M0 3v18h24v-18h-24zm6.623 7.929l-4.623 5.712v-9.458l4.623 3.746zm-4.141-5.929h19.035l-9.517 7.713-9.518-7.713zm5.694 7.188l3.824 3.099 3.83-3.104 5.612 6.817h-18.779l5.513-6.812zm9.208-1.264l4.616-3.741v9.348l-4.616-5.607z' />
							</svg>
						</a>
						<a
							href={TWITTER_URL}
							target='_blank'
							rel='noopener noreferrer'
							className='social-link'
							title='Twitter'
						>
							<svg viewBox='0 0 24 24' className='icon twitter-icon'>
								<path d='M24 4.557c-.883.392-1.832.656-2.828.775 1.017-.609 1.798-1.574 2.165-2.724-.951.564-2.005.974-3.127 1.195-.897-.957-2.178-1.555-3.594-1.555-3.179 0-5.515 2.966-4.797 6.045-4.091-.205-7.719-2.165-10.148-5.144-1.29 2.213-.669 5.108 1.523 6.574-.806-.026-1.566-.247-2.229-.616-.054 2.281 1.581 4.415 3.949 4.89-.693.188-1.452.232-2.224.084.626 1.956 2.444 3.379 4.6 3.419-2.07 1.623-4.678 2.348-7.29 2.04 2.179 1.397 4.768 2.212 7.548 2.212 9.142 0 14.307-7.721 13.995-14.646.962-.695 1.797-1.562 2.457-2.549z' />
							</svg>
						</a>
					</div>
					<div className='contributor'>
						<span className='contributor-name'>Somitav:</span>
						<a
							href={SOMITAV_GITHUB_URL}
							target='_blank'
							rel='noopener noreferrer'
							className='social-link'
							title='GitHub'
						>
							<svg viewBox='0 0 24 24' className='icon github-icon'>
								<path d='M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z' />
							</svg>
						</a>
						<a
							href={SOMITAV_LINKEDIN_URL}
							target='_blank'
							rel='noopener noreferrer'
							className='social-link'
							title='LinkedIn'
						>
							<svg viewBox='0 0 24 24' className='icon linkedin-icon'>
								<path d='M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z' />
							</svg>
						</a>
						<a
							href={`mailto:${SOMITAV_GMAIL_URL}`}
							className='social-link'
							title='Email'
						>
							<svg viewBox='0 0 24 24' className='icon email-icon'>
								<path d='M0 3v18h24v-18h-24zm6.623 7.929l-4.623 5.712v-9.458l4.623 3.746zm-4.141-5.929h19.035l-9.517 7.713-9.518-7.713zm5.694 7.188l3.824 3.099 3.83-3.104 5.612 6.817h-18.779l5.513-6.812zm9.208-1.264l4.616-3.741v9.348l-4.616-5.607z' />
							</svg>
						</a>
					</div>
				</div>
			</footer>

			<button className='help-button' aria-label='Help'>
				?
			</button>
		</div>
	);
}

export default App;
