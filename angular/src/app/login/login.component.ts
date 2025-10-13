import { Component } from '@angular/core';
import { LoginService } from '../services/login.service';
import { Router } from '@angular/router';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [],
  templateUrl: './login.component.html',
  styleUrl: './login.component.scss'
})
export class LoginComponent {
user: any = null;

  constructor(private loginService: LoginService,
  private  router: Router
  ) {}

  async login() {
    this.user = await this.loginService.loginWithGoogle();
    console.log('Logged in user:', this.user);

    if (this.user) {
      this.router.navigateByUrl('/dashboard');   // 👈 works after login
    }
  }


}
