import { inject, Injectable } from '@angular/core';
import { getAuth, Auth, GoogleAuthProvider, signInWithPopup, signOut, User } from '@angular/fire/auth';
import { ToastrService } from 'ngx-toastr';


@Injectable({
  providedIn: 'root'
})
export class LoginService {
  private tokenKey = 'authtoken';
  private toastr = inject(ToastrService);

  constructor(private auth: Auth) { }

  async loginWithGoogle(): Promise<User | null> {
    const provider = new GoogleAuthProvider();
  return await signInWithPopup(this.auth, provider)
      .then((result) => {
        result.user.getIdTokenResult().then((data) => {
          sessionStorage.setItem(this.tokenKey, data.token);
        });
        return result.user;
      })
      .catch((error) => {
        // Handle Errors here.
        const errorCode = error.code;
        if (errorCode === 'auth/account-exists-with-different-credential') {
          this.toastr.error('You have already signed up with a different auth provider for that email.', 'Heads up!', {
            timeOut: 6000,
            positionClass: 'toast-bottom-center'
          });
          console.error('You have already signed up with a different auth provider for that email.');
        } else {
          console.error(error.message);
        }
        throw error; // Re-throw to ensure the promise rejects and does not return void
      });
  }

 googlesignOut(): Promise<void> {
    const auth = getAuth();
    return signOut(auth).then(() => {
      sessionStorage.removeItem(this.tokenKey);
    }).catch((err) => {
      console.error("Sign out error", err);
      throw err;
    });
  }


  get currentUser() {
    return this.auth.currentUser;
  }
}
