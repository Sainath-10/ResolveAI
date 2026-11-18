# Technical Troubleshooting and Account Access

## Login loop
If a customer is signed in but immediately returned to the login page:
1. Ask them to clear cookies for the storefront domain.
2. Ask them to try a private/incognito window.
3. Confirm the email address has been verified.
4. Check the service status for an active login incident.

## Locked account
Accounts lock after five failed sign-in attempts. A password reset does not unlock the account by itself. Support can confirm the lock state through the customer profile. Unlocking requires identity verification and is performed by a human agent.

## Password reset not arriving
Ask the customer to check spam, confirm the email on file, and wait up to 10 minutes. Support must not change an account email without identity verification.

## Two-factor authentication
If the authenticator device is lost, recovery requires identity verification and human review. Support cannot bypass two-factor authentication.

## App or page not loading
Ask the customer to clear the browser cache, disable extensions, try another browser, and check the service status page. If other customers report the same problem, treat it as a platform incident.

## API rate limits
Free plans allow 60 requests per minute; paid plans allow 600. A 429 response includes a `Retry-After` header. Clients should back off for the indicated number of seconds.

## Payment failures
A declined card is usually resolved by updating the payment method. Support cannot retry or charge a card manually. Repeated declines should be escalated if the customer reports being charged.

## When to escalate
Escalate when identity verification is required, when a policy exception is requested, when the account is locked, or when troubleshooting steps have already failed.
